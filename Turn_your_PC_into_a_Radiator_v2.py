"""
Charge PC - deux potentiometres (CPU / GPU) avec asservissement reel
----------------------------------------------------------------------
Chaque cadran pilote l'utilisation reelle du composant correspondant en
boucle fermee, affiche le % reel et la temperature, et coupe automatiquement
la charge si la temperature depasse un seuil de securite.

Dependances :
    pip install psutil WinTmp
Optionnel (mesure/charge GPU reelle, NVIDIA uniquement) :
    pip install pynvml moderngl

IMPORTANT : WinTmp (lecture des temperatures) a besoin des droits
administrateur pour acceder aux capteurs materiels. Lance une invite de
commandes EN ADMINISTRATEUR avant de lancer le script :
    python charge_pc_v2.py

Pour en faire un .exe (a executer avec PyInstaller sur la machine Windows
cible elle-meme) :
    pip install pyinstaller
    pyinstaller --onefile --uac-admin --windowed --name ChargePC charge_pc_v2.py
    (--uac-admin fait que l'exe demande automatiquement les droits admin)
"""

import math
import time
import warnings
import threading
import multiprocessing as mp
import tkinter as tk

import psutil

# Fait taire l'avertissement de depreciation de pynvml (le paquet fonctionne
# encore tres bien, l'avertissement recommande juste de migrer vers le
# paquet "nvidia-ml-py" a terme -> sans incidence ici).
warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")

# ---------------------------------------------------------------------------
# Seuil de securite thermique (modifiable ici)
# ---------------------------------------------------------------------------
SAFE_TEMP_C = 85.0       # au-dela : charge coupee automatiquement
SAFE_TEMP_HYSTERESIS = 5.0  # ne reprend qu'en dessous de (SAFE_TEMP_C - hysteresis)

# ---------------------------------------------------------------------------
# Dependances optionnelles
# ---------------------------------------------------------------------------
GPU_MONITOR_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_MONITOR_AVAILABLE = True
except Exception:
    GPU_MONITOR_AVAILABLE = False

GPU_RENDER_AVAILABLE = False
try:
    import moderngl
    GPU_RENDER_AVAILABLE = True
except Exception:
    GPU_RENDER_AVAILABLE = False

TEMP_AVAILABLE = False
TEMP_IMPORT_ERROR = ""
_hw_computer = None
try:
    from HardwareMonitor.Hardware import Computer as _HwComputer, HardwareType as _HwType
    _hw_computer = _HwComputer()
    # On ne garde que CPU + GPU : le module Memoire (RAM SPD) plante a
    # l'ouverture sur certaines machines (assembly manquante), et ca fait
    # tomber tout l'objet Computer avec lui si on le laisse active.
    _hw_computer.IsCpuEnabled = True
    _hw_computer.IsGpuEnabled = True
    _hw_computer.IsMemoryEnabled = False
    _hw_computer.IsMotherboardEnabled = False
    _hw_computer.IsStorageEnabled = False
    _hw_computer.IsNetworkEnabled = False
    _hw_computer.IsControllerEnabled = False
    _hw_computer.IsBatteryEnabled = False
    _hw_computer.Open()
    TEMP_AVAILABLE = True
except Exception as exc:
    TEMP_AVAILABLE = False
    TEMP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _read_max_temp(hardware_type):
    """Renvoie la temperature max relevee pour un type de materiel donne
    (ex. HardwareType.Cpu), ou None si aucun capteur trouve."""
    if _hw_computer is None:
        return None
    best = None
    for hw in _hw_computer.Hardware:
        hw.Update()
        items = [hw] if hw.HardwareType == hardware_type else []
        for sub in hw.SubHardware:
            sub.Update()
            if sub.HardwareType == hardware_type:
                items.append(sub)
        for item in items:
            for sensor in item.Sensors:
                if str(sensor.SensorType) == "Temperature" and sensor.Value is not None:
                    v = float(sensor.Value)
                    if best is None or v > best:
                        best = v
    return best


# ---------------------------------------------------------------------------
# Travailleur CPU (processus separe -> vrai coeur charge)
# ---------------------------------------------------------------------------
def cpu_worker(duty_value, stop_flag, worker_index=0, worker_count=1):
    cycle = 0.05
    # Decale la phase de chaque travailleur pour que les coeurs ne boostent
    # pas tous au maximum en meme temps (evite les pics de temperature
    # inutiles pour une meme charge moyenne).
    phase_offset = cycle * (worker_index / max(1, worker_count))
    time.sleep(phase_offset)
    while not stop_flag.value:
        duty = duty_value.value
        busy_time = cycle * (duty / 100.0)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < busy_time:
            pass
        rest = cycle - busy_time
        if rest > 0:
            time.sleep(rest)


class CpuController(threading.Thread):
    """Lit l'utilisation CPU reelle et corrige le duty-cycle des travailleurs."""

    def __init__(self, get_target, duty_value, on_actual_update):
        super().__init__(daemon=True)
        self.get_target = get_target
        self.duty_value = duty_value
        self.on_actual_update = on_actual_update
        self.running = True
        self._kp = 0.7

    def run(self):
        psutil.cpu_percent(interval=None)
        while self.running:
            actual = psutil.cpu_percent(interval=0.5)
            target = self.get_target()
            if target <= 0:
                self.duty_value.value = 0.0
            else:
                error = target - actual
                new_duty = self.duty_value.value + self._kp * error
                self.duty_value.value = max(0.0, min(100.0, new_duty))
            self.on_actual_update(actual)

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# GPU : rendu + asservissement dans un PROCESSUS separe
# ---------------------------------------------------------------------------
VERTEX_SHADER = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAGMENT_SHADER = """
#version 330
uniform int iterations;
uniform float seed;
out vec4 f_color;
void main() {
    vec2 c = gl_FragCoord.xy * 0.001 + seed;
    vec2 z = vec2(0.0);
    float m = 0.0;
    for (int i = 0; i < 4000; i++) {
        if (i >= iterations) break;
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        m += dot(z, z) * 0.00001;
    }
    f_color = vec4(m, m, m, 1.0);
}
"""

GPU_STATUS_TEXT = {
    0: "GPU : initialisation...",
    1: "GPU : moderngl absent, charge GPU desactivee",
    2: "GPU : contexte graphique indisponible",
    3: "GPU : mesure reelle active (NVIDIA)",
    4: "GPU : pas de mesure reelle (boucle ouverte)",
}


def gpu_worker_process(target_value, actual_value, status_code, stop_flag):
    if not GPU_RENDER_AVAILABLE:
        status_code.value = 1
        return
    try:
        ctx = moderngl.create_context(standalone=True, require=330)
        prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        vbo = ctx.buffer(
            data=bytes(bytearray(__import__("struct").pack("8f", -1, -1, 1, -1, -1, 1, 1, 1)))
        )
        vao = ctx.simple_vertex_array(prog, vbo, "in_pos")
        fbo = ctx.simple_framebuffer((256, 256))
        fbo.use()
    except Exception:
        status_code.value = 2
        return

    monitor_available = False
    gpu_handle = None
    if GPU_MONITOR_AVAILABLE:
        try:
            import pynvml as _pynvml
            _pynvml.nvmlInit()
            gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            monitor_available = True
        except Exception:
            monitor_available = False

    status_code.value = 3 if monitor_available else 4

    cycle = 0.08
    iterations = 350
    duty = 0.0
    kp = 0.5
    ki = 0.35
    integral = 0.0
    last_sample = 0.0
    sample_interval = 1.0  # cadence de rafraichissement du pilote NVIDIA

    while not stop_flag.value:
        target = target_value.value
        now = time.perf_counter()

        if monitor_available:
            if now - last_sample >= sample_interval:
                import pynvml as _pynvml
                util = _pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                actual_value.value = float(util)
                if target <= 0:
                    duty = 0.0
                    integral = 0.0
                else:
                    error = target - util
                    integral = max(-150.0, min(150.0, integral + error))
                    duty = max(0.0, min(100.0, duty + kp * error * 0.3 + ki * integral * 0.05))
                last_sample = now
        else:
            actual_value.value = -1.0
            duty = target

        busy_time = cycle * (duty / 100.0)
        if busy_time > 0:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < busy_time:
                prog["iterations"].value = iterations
                prog["seed"].value = (time.time() * 0.001) % 1.0
                vao.render(moderngl.TRIANGLE_STRIP)
                ctx.finish()
            rest = cycle - busy_time
        else:
            rest = cycle
        if rest > 0:
            time.sleep(rest)


# ---------------------------------------------------------------------------
# Temperatures + securite thermique (thread, process principal)
# ---------------------------------------------------------------------------
class TempMonitor(threading.Thread):
    def __init__(self, on_update, get_cpu_limit, get_gpu_limit):
        super().__init__(daemon=True)
        self.on_update = on_update
        self.get_cpu_limit = get_cpu_limit
        self.get_gpu_limit = get_gpu_limit
        self.running = True
        self.cpu_tripped = False
        self.gpu_tripped = False

    def run(self):
        if not TEMP_AVAILABLE:
            msg = f"Températures indisponibles : {TEMP_IMPORT_ERROR}" if TEMP_IMPORT_ERROR else "pip install HardwareMonitor (+ lancer en administrateur)"
            self.on_update(None, None, False, False, msg)
            return
        while self.running:
            err = ""
            try:
                cpu_t = _read_max_temp(_HwType.Cpu)
            except Exception as exc:
                cpu_t = None
                err = f"CPU temp : {exc}"
            try:
                gpu_t = _read_max_temp(_HwType.GpuNvidia)
            except Exception as exc:
                gpu_t = None
                err = err or f"GPU temp : {exc}"

            cpu_limit = self.get_cpu_limit()
            gpu_limit = self.get_gpu_limit()
            if cpu_t is not None:
                if cpu_t >= cpu_limit:
                    self.cpu_tripped = True
                elif cpu_t < cpu_limit - SAFE_TEMP_HYSTERESIS:
                    self.cpu_tripped = False
            if gpu_t is not None:
                if gpu_t >= gpu_limit:
                    self.gpu_tripped = True
                elif gpu_t < gpu_limit - SAFE_TEMP_HYSTERESIS:
                    self.gpu_tripped = False

            self.on_update(cpu_t, gpu_t, self.cpu_tripped, self.gpu_tripped, err)
            time.sleep(1.5)

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# Widget cadran reutilisable
# ---------------------------------------------------------------------------
class Dial(tk.Canvas):
    SIZE = 170
    CENTER = SIZE / 2
    RADIUS = 68
    START_DEG = 135
    SWEEP_DEG = 270

    def __init__(self, parent, accent, on_change, **kw):
        super().__init__(parent, width=self.SIZE, height=self.SIZE,
                          bg=kw.pop("bg", "#141414"), highlightthickness=0, **kw)
        self.accent = accent
        self.on_change = on_change
        self.percent = 0.0
        self.bind("<Button-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)
        self._draw()

    def _point(self, clock_deg, radius):
        a = math.radians(clock_deg - 90)
        return self.CENTER + radius * math.cos(a), self.CENTER + radius * math.sin(a)

    def _arc(self, from_deg, to_deg, color, width=9):
        steps = 36
        pts = [self._point(from_deg + (to_deg - from_deg) * (i / steps), self.RADIUS) for i in range(steps + 1)]
        for i in range(len(pts) - 1):
            self.create_line(*pts[i], *pts[i + 1], fill=color, width=width, capstyle=tk.ROUND)

    def _draw(self):
        self.delete("all")
        self._arc(self.START_DEG, self.START_DEG + self.SWEEP_DEG, "#2a2a2a")
        if self.percent > 0:
            end = self.START_DEG + (self.percent / 100.0) * self.SWEEP_DEG
            self._arc(self.START_DEG, end, self.accent)
        self.create_oval(self.CENTER - 48, self.CENTER - 48, self.CENTER + 48, self.CENTER + 48,
                          fill="#1f1f1f", outline="#0a0a0a", width=2)
        clock_deg = self.START_DEG + (self.percent / 100.0) * self.SWEEP_DEG
        x2, y2 = self._point(clock_deg, 38)
        self.create_line(self.CENTER, self.CENTER, x2, y2, fill=self.accent, width=4, capstyle=tk.ROUND)
        self.create_text(self.CENTER, self.CENTER, text=f"{round(self.percent)}%",
                          fill="#eae6e0", font=("Segoe UI", 18, "bold"))

    def _on_drag(self, event):
        dx, dy = event.x - self.CENTER, event.y - self.CENTER
        raw = math.degrees(math.atan2(dy, dx))
        clock_deg = (raw + 90 + 360) % 360
        end_deg = self.START_DEG + self.SWEEP_DEG
        cont = clock_deg + 360 if clock_deg < self.START_DEG else clock_deg
        if cont > end_deg:
            dist_end = cont - end_deg
            dist_start = (self.START_DEG + 360) - cont
            cont = end_deg if dist_end < dist_start else self.START_DEG
        pct = (cont - self.START_DEG) / self.SWEEP_DEG * 100.0
        self.percent = max(0.0, min(100.0, pct))
        self._draw()
        self.on_change(self.percent)


# ---------------------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------------------
class App:
    BG = "#141414"
    TEXT = "#eae6e0"
    TEXT_DIM = "#8a8580"
    WARN = "#ff4444"
    CPU_ACCENT = "#ff7a1a"
    GPU_ACCENT = "#2ec4ff"

    def __init__(self, root):
        self.root = root
        root.title("Charge PC")
        root.configure(bg=self.BG)
        root.resizable(False, False)

        self.cpu_target = 0.0
        self.gpu_target = 0.0
        self.cpu_actual = 0.0
        self.cpu_temp = None
        self.gpu_temp = None
        self.cpu_tripped = False
        self.gpu_tripped = False
        self.temp_error = ""
        self.cpu_safe_temp = SAFE_TEMP_C
        self.gpu_safe_temp = SAFE_TEMP_C

        cols = tk.Frame(root, bg=self.BG)
        cols.pack(padx=16, pady=16)

        cpu_col = tk.Frame(cols, bg=self.BG)
        gpu_col = tk.Frame(cols, bg=self.BG)
        cpu_col.grid(row=0, column=0, padx=14)
        gpu_col.grid(row=0, column=1, padx=14)

        tk.Label(cpu_col, text="CPU", font=("Segoe UI", 10, "bold"),
                 fg=self.TEXT_DIM, bg=self.BG).pack()
        self.cpu_dial = Dial(cpu_col, self.CPU_ACCENT, self._on_cpu_change, bg=self.BG)
        self.cpu_dial.pack(pady=(4, 8))
        self.cpu_sub = tk.Label(cpu_col, text="réel: -- %   temp: --°C",
                                 font=("Segoe UI", 9), fg=self.TEXT_DIM, bg=self.BG)
        self.cpu_sub.pack()
        self.cpu_limit_entry = self._make_limit_row(cpu_col, self.cpu_safe_temp, self._update_cpu_limit)
        self.cpu_warn = tk.Label(cpu_col, text="", font=("Segoe UI", 8, "bold"),
                                  fg=self.WARN, bg=self.BG)
        self.cpu_warn.pack()

        tk.Label(gpu_col, text="GPU", font=("Segoe UI", 10, "bold"),
                 fg=self.TEXT_DIM, bg=self.BG).pack()
        self.gpu_dial = Dial(gpu_col, self.GPU_ACCENT, self._on_gpu_change, bg=self.BG)
        self.gpu_dial.pack(pady=(4, 8))
        self.gpu_sub = tk.Label(gpu_col, text="réel: -- %   temp: --°C",
                                 font=("Segoe UI", 9), fg=self.TEXT_DIM, bg=self.BG)
        self.gpu_sub.pack()
        self.gpu_limit_entry = self._make_limit_row(gpu_col, self.gpu_safe_temp, self._update_gpu_limit)
        self.gpu_warn = tk.Label(gpu_col, text="", font=("Segoe UI", 8, "bold"),
                                  fg=self.WARN, bg=self.BG)
        self.gpu_warn.pack()

        stop_btn = tk.Button(
            root, text="⏹  Tout arrêter", command=self._stop_all,
            bg="#3a1414", fg=self.TEXT, activebackground="#5a1c1c", activeforeground=self.TEXT,
            relief="flat", font=("Segoe UI", 9, "bold"), padx=14, pady=5, bd=0,
        )
        stop_btn.pack(pady=(2, 8))

        self.status = tk.Label(root, text="", font=("Segoe UI", 8),
                                fg=self.TEXT_DIM, bg=self.BG)
        self.status.pack(pady=(0, 10))

        # --- CPU backend ---
        self.cores = mp.cpu_count() or 4
        self.duty_value = mp.Value("d", 0.0, lock=False)
        self.stop_flag = mp.Value("i", 0, lock=False)
        self.workers = [
            mp.Process(target=cpu_worker, args=(self.duty_value, self.stop_flag, i, self.cores), daemon=True)
            for i in range(self.cores)
        ]
        for w in self.workers:
            w.start()
        self.cpu_ctrl = CpuController(
            get_target=lambda: 0.0 if self.cpu_tripped else self.cpu_target,
            duty_value=self.duty_value,
            on_actual_update=self._on_cpu_actual,
        )
        self.cpu_ctrl.start()

        # --- GPU backend ---
        self.gpu_target_value = mp.Value("d", 0.0, lock=False)
        self.gpu_actual_value = mp.Value("d", -1.0, lock=False)
        self.gpu_status_code = mp.Value("i", 0, lock=False)
        self.gpu_stop_flag = mp.Value("i", 0, lock=False)
        self.gpu_process = mp.Process(
            target=gpu_worker_process,
            args=(self.gpu_target_value, self.gpu_actual_value, self.gpu_status_code, self.gpu_stop_flag),
            daemon=True,
        )
        self.gpu_process.start()

        # --- Temperatures + securite ---
        self.temp_monitor = TempMonitor(
            on_update=self._on_temp_update,
            get_cpu_limit=lambda: self.cpu_safe_temp,
            get_gpu_limit=lambda: self.gpu_safe_temp,
        )
        self.temp_monitor.start()
        if not TEMP_AVAILABLE:
            self.status.config(text="Températures indisponibles : pip install WinTmp (+ lancer en administrateur)")

        self._tick_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- callbacks cadrans --------------------------------------------------
    # -- reglage des seuils de securite / bouton d'arret --------------------
    def _make_limit_row(self, parent, initial_value, on_change):
        row = tk.Frame(parent, bg=self.BG)
        row.pack(pady=(4, 0))
        tk.Label(row, text="seuil sécu:", font=("Segoe UI", 8),
                 fg=self.TEXT_DIM, bg=self.BG).pack(side="left")
        entry = tk.Entry(row, width=4, justify="center", relief="flat",
                          bg="#1f1f1f", fg=self.TEXT, insertbackground=self.TEXT)
        entry.insert(0, str(int(initial_value)))
        entry.pack(side="left", padx=(4, 2))
        tk.Label(row, text="°C", font=("Segoe UI", 8),
                 fg=self.TEXT_DIM, bg=self.BG).pack(side="left")
        entry.bind("<Return>", lambda e: on_change(entry))
        entry.bind("<FocusOut>", lambda e: on_change(entry))
        return entry

    def _parse_limit(self, entry, current_value):
        try:
            val = float(entry.get())
            val = max(40.0, min(100.0, val))
        except ValueError:
            val = current_value
        entry.delete(0, tk.END)
        entry.insert(0, str(int(val)))
        return val

    def _update_cpu_limit(self, entry):
        self.cpu_safe_temp = self._parse_limit(entry, self.cpu_safe_temp)

    def _update_gpu_limit(self, entry):
        self.gpu_safe_temp = self._parse_limit(entry, self.gpu_safe_temp)

    def _stop_all(self):
        self.cpu_target = 0.0
        self.gpu_target = 0.0
        self.cpu_dial.percent = 0.0
        self.cpu_dial._draw()
        self.gpu_dial.percent = 0.0
        self.gpu_dial._draw()

    # -- callbacks cadrans --------------------------------------------------
    def _on_cpu_change(self, pct):
        self.cpu_target = pct

    def _on_gpu_change(self, pct):
        self.gpu_target = pct

    # -- callbacks asservissement --------------------------------------------------
    def _on_cpu_actual(self, value):
        self.cpu_actual = value

    def _on_temp_update(self, cpu_t, gpu_t, cpu_tripped, gpu_tripped, err=""):
        self.cpu_temp = cpu_t
        self.gpu_temp = gpu_t
        self.cpu_tripped = cpu_tripped
        self.gpu_tripped = gpu_tripped
        self.temp_error = err

    # -- boucle d'affichage --------------------------------------------------
    def _tick_ui(self):
        cpu_t_txt = f"{round(self.cpu_temp)}°C" if self.cpu_temp is not None else "--°C"
        self.cpu_sub.config(text=f"réel: {round(self.cpu_actual)}%   temp: {cpu_t_txt}")
        self.cpu_warn.config(
            text=f"⚠ sécurité : charge coupée ({self.cpu_safe_temp:.0f}°C)" if self.cpu_tripped else ""
        )

        gpu_actual = self.gpu_actual_value.value
        gpu_actual_txt = f"{round(gpu_actual)}%" if gpu_actual >= 0 else "--"
        gpu_t_txt = f"{round(self.gpu_temp)}°C" if self.gpu_temp is not None else "--°C"
        self.gpu_sub.config(text=f"réel: {gpu_actual_txt}   temp: {gpu_t_txt}")
        self.gpu_warn.config(
            text=f"⚠ sécurité : charge coupée ({self.gpu_safe_temp:.0f}°C)" if self.gpu_tripped else ""
        )

        # applique la cible GPU (avec coupure de securite) au processus GPU
        self.gpu_target_value.value = 0.0 if self.gpu_tripped else self.gpu_target

        if self.temp_error:
            self.status.config(text=f"Temp : {self.temp_error}")
        elif TEMP_AVAILABLE:
            self.status.config(text=GPU_STATUS_TEXT.get(self.gpu_status_code.value, ""))

        self.root.after(300, self._tick_ui)

    def _on_close(self):
        self.cpu_ctrl.stop()
        self.temp_monitor.stop()
        self.stop_flag.value = 1
        self.gpu_stop_flag.value = 1
        for w in self.workers:
            w.terminate()
        if self.gpu_process.is_alive():
            self.gpu_process.terminate()
        self.root.destroy()


if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    App(root)
    root.mainloop()
