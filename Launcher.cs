﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Windows.Forms;

// ViralForge desktop launcher (friend-PC ready).
// - Finds Python dynamically (bundled venv/python, common installs, PATH, py.exe)
// - Ensures bundled bin/ffmpeg is on PATH for child processes
// - First-run bootstrap: creates/uses local .venv and installs requirements if needed
// - Starts backend hidden, waits for health, opens native window
// - No hardcoded %USERPROFILE% paths

class Program
{
    private static readonly string AppDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
    private static readonly string DataDir = Path.Combine(AppDir, "data");
    private static readonly string LogDir = Path.Combine(DataDir, "logs");
    private static readonly string LogFilePath = Path.Combine(DataDir, "launcher.log");
    private static readonly string RequirementsPath = Path.Combine(AppDir, "requirements.txt");
    private static readonly string HealthCheckUrl = "http://127.0.0.1:8821/api/system/health";
    private static readonly string WebUiUrl = "http://127.0.0.1:8821";

    [STAThread]
    static void Main(string[] args)
    {
        try
        {
            Directory.SetCurrentDirectory(AppDir);
            Directory.CreateDirectory(DataDir);
            Directory.CreateDirectory(LogDir);
            Log("ViralForge launcher started");
            Log("AppDir=" + AppDir);

            if (IsServerRunning())
            {
                Log("Backend already running; opening UI");
                var pyUi = ResolvePython(out var pywUi, out var usePy);
                if (string.IsNullOrWhiteSpace(pyUi)) throw new Exception("Python not found to open UI window.");
                OpenUiWindow(pywUi ?? pyUi, usePy);
                return;
            }

            var python = ResolvePython(out var pythonw, out var usedPyLauncher);
            if (string.IsNullOrWhiteSpace(python))
            {
                TryInstallPython();
                python = ResolvePython(out pythonw, out usedPyLauncher);
            }
            if (string.IsNullOrWhiteSpace(python))
            {
                throw new Exception(
                    "Python 3.10+ is required but was not found.\n\n" +
                    "Install Python 3.12 from python.org (or Microsoft Store/winget), then run ViralForge again.\n" +
                    "Tip: winget install -e --id Python.Python.3.12");
            }
            Log("Using Python: " + python + (usedPyLauncher ? " (py launcher)" : ""));

            python = EnsureDependencies(python, pythonw, usedPyLauncher);
            pythonw = pythonw ?? python;

            StartServer(python, usedPyLauncher);
            if (!WaitForServerReady())
            {
                throw new Exception("The backend did not start in time. See " + LogFilePath);
            }
            OpenUiWindow(pythonw, usedPyLauncher);
        }
        catch (Exception ex)
        {
            try { Log("ERROR: " + ex); } catch { }
            try
            {
                MessageBox.Show(
                    "ViralForge failed to start:\n\n" + ex.Message + "\n\nLog: " + LogFilePath,
                    "ViralForge",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
            catch { }
        }
    }

    static void Log(string message)
    {
        try
        {
            var line = "[" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + "] " + message + Environment.NewLine;
            File.AppendAllText(LogFilePath, line);
        }
        catch { }
    }

    static string ResolvePython(out string pythonw, out bool usePyLauncher)
    {
        pythonw = null; usePyLauncher = false;
        var candidates = new List<string>();
        // 1) Bundled/local envs inside app folder
        candidates.Add(Path.Combine(AppDir, ".venv", "Scripts", "python.exe"));
        candidates.Add(Path.Combine(AppDir, "venv", "Scripts", "python.exe"));
        candidates.Add(Path.Combine(AppDir, "python", "python.exe"));
        candidates.Add(Path.Combine(AppDir, "runtime", "python", "python.exe"));
        // 2) Common per-user/system installs
        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var roots = new[] {
            Path.Combine(local, "Programs", "Python"),
            @"C:\Program Files",
            @"C:\Program Files (x86)"
        };
        foreach (var r in roots)
        {
            try
            {
                if (!Directory.Exists(r)) continue;
                foreach (var d in Directory.GetDirectories(r, "Python3*").OrderByDescending(x => x))
                {
                    candidates.Add(Path.Combine(d, "python.exe"));
                }
            }
            catch { }
        }
        // 3) PATH
        candidates.Add(FindOnPath("python.exe"));
        candidates.Add(FindOnPath("python3.exe"));

        foreach (var c in candidates.Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                if (!File.Exists(c)) continue;
                if (IsSupportedPython(c, false)) { pythonw = GuessPythonw(c); return c; }
            }
            catch { }
        }
        // 4) py launcher
        var py = FindOnPath("py.exe");
        if (!string.IsNullOrWhiteSpace(py) && IsSupportedPython(py, true))
        {
            usePyLauncher = true;
            pythonw = py; // we'll pass -3 -m ... and for window try pythonw via pyw if exists
            var pyw = FindOnPath("pyw.exe");
            if (!string.IsNullOrWhiteSpace(pyw)) pythonw = pyw;
            return py;
        }
        return null;
    }

    static bool IsSupportedPython(string exe, bool isPy)
    {
        try
        {
            var args = isPy ? "-3 -c \"import sys; v=sys.version_info; print(v.major, v.minor)\"" : "-c \"import sys; v=sys.version_info; print(v.major, v.minor)\"";
            var psi = new ProcessStartInfo(exe, args) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true, WorkingDirectory = AppDir };
            using (var p = Process.Start(psi))
            {
                if (p == null) return false;
                if (!p.WaitForExit(8000)) { try { p.Kill(); } catch { } return false; }
                var outp = (p.StandardOutput.ReadToEnd() ?? "").Trim();
                var parts = outp.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2) return p.ExitCode == 0; // fallback
                int maj, min;
                if (int.TryParse(parts[0], out maj) && int.TryParse(parts[1], out min))
                    return maj == 3 && min >= 10 && min <= 13;
                return p.ExitCode == 0;
            }
        }
        catch { return false; }
    }

    static string GuessPythonw(string pythonExe)
    {
        try
        {
            var dir = Path.GetDirectoryName(pythonExe);
            var w = Path.Combine(dir ?? "", "pythonw.exe");
            if (File.Exists(w)) return w;
        }
        catch { }
        return pythonExe;
    }

    static string FindOnPath(string name)
    {
        try
        {
            var path = Environment.GetEnvironmentVariable("PATH") ?? "";
            foreach (var p in path.Split(Path.PathSeparator))
            {
                try
                {
                    var c = Path.Combine(p.Trim(), name);
                    if (File.Exists(c)) return c;
                }
                catch { }
            }
        }
        catch { }
        return null;
    }

    static void TryInstallPython()
    {
        try
        {
            var winget = FindOnPath("winget.exe");
            if (string.IsNullOrWhiteSpace(winget)) return;
            var res = MessageBox.Show(
                "Python 3.10+ is required for ViralForge.\n\nClick Yes to install Python 3.12 automatically (winget). Click No to install manually.",
                "ViralForge setup",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Information);
            if (res != DialogResult.Yes) return;
            Log("Attempting winget install of Python 3.12");
            var psi = new ProcessStartInfo(winget, "install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements")
            { UseShellExecute = false, CreateNoWindow = false, WorkingDirectory = AppDir };
            using (var p = Process.Start(psi)) { if (p != null) p.WaitForExit(600000); }
        }
        catch (Exception ex) { Log("winget install failed: " + ex.Message); }
    }

    static string EnsureDependencies(string python, string pythonw, bool usePyLauncher)
    {
        // Prefer local .venv for friend PCs to avoid global permission issues.
        var venvPy = Path.Combine(AppDir, ".venv", "Scripts", "python.exe");
        var venvPyw = Path.Combine(AppDir, ".venv", "Scripts", "pythonw.exe");
        try
        {
            if (!File.Exists(venvPy))
            {
                Log("Creating local .venv for first-run bootstrap");
                RunPython(python, usePyLauncher, "-m venv .venv", wait: true, showWindow: true, timeoutMs: 600000);
            }
        }
        catch (Exception ex) { Log("venv create failed (continuing with selected python): " + ex.Message); }

        var runPy = File.Exists(venvPy) ? venvPy : python;
        var runPyw = File.Exists(venvPyw) ? venvPyw : (pythonw ?? runPy);
        bool runIsVenv = File.Exists(venvPy) && string.Equals(runPy, venvPy, StringComparison.OrdinalIgnoreCase);

        try
        {
            // quick check for core deps
            int rc = RunPython(runPy, (!runIsVenv && usePyLauncher), "-c \"import fastapi,uvicorn\" ", wait: true, showWindow: false, timeoutMs: 60000, throwOnFail: false);
            if (rc != 0 && File.Exists(RequirementsPath))
            {
                Log("Installing requirements (first run)...");
                // show window so friend sees progress
                RunPython(runPy, (!runIsVenv && usePyLauncher), "-m pip install --upgrade pip", wait: true, showWindow: true, timeoutMs: 600000, throwOnFail: false);
                RunPython(runPy, (!runIsVenv && usePyLauncher), "-m pip install -r requirements.txt", wait: true, showWindow: true, timeoutMs: 1800000, throwOnFail: false);
            }
        }
        catch (Exception ex) { Log("dependency bootstrap warning: " + ex.Message); }

        return runPy; // window will derive pythonw from this
    }

    static int RunPython(string python, bool usePyLauncher, string args, bool wait, bool showWindow, int timeoutMs, bool throwOnFail = true)
    {
        var psi = new ProcessStartInfo();
        if (usePyLauncher)
        {
            psi.FileName = python; // py.exe or pyw.exe
            psi.Arguments = "-3 " + args;
        }
        else
        {
            psi.FileName = python;
            psi.Arguments = args;
        }
        psi.WorkingDirectory = AppDir;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = !showWindow;
        ApplyChildEnv(psi);
        var p = Process.Start(psi);
        if (p == null) { if (throwOnFail) throw new Exception("Failed to start: " + psi.FileName); return -1; }
        if (!wait) return 0;
        if (!p.WaitForExit(timeoutMs)) { try { p.Kill(); } catch { } if (throwOnFail) throw new Exception("Process timed out"); return -1; }
        if (throwOnFail && p.ExitCode != 0) throw new Exception("Command failed (" + p.ExitCode + "): " + psi.FileName + " " + psi.Arguments);
        return p.ExitCode;
    }

    static void StartServer(string python, bool usePyLauncher)
    {
        var psi = new ProcessStartInfo();
        if (usePyLauncher)
        {
            psi.FileName = python; // py.exe
            psi.Arguments = "-3 -m clipforge --host 127.0.0.1 --port 8821 --no-window";
        }
        else
        {
            psi.FileName = python;
            psi.Arguments = "-m clipforge --host 127.0.0.1 --port 8821 --no-window";
        }
        psi.WorkingDirectory = AppDir;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        ApplyChildEnv(psi);
        psi.EnvironmentVariables["CLIPFORGE_NO_TOKEN"] = "1";
        var p = Process.Start(psi);
        Log("Backend started PID=" + (p != null ? p.Id.ToString() : "?") + " using " + psi.FileName);
    }

    static void OpenUiWindow(string pythonOrPyw, bool usePyLauncher)
    {
        var exe = pythonOrPyw;
        var args = "-m clipforge.native_window";
        try
        {
            if (usePyLauncher)
            {
                var pyw = FindOnPath("pyw.exe");
                if (!string.IsNullOrWhiteSpace(pyw)) { exe = pyw; args = "-3 -m clipforge.native_window"; }
                else { exe = pythonOrPyw; args = "-3 -m clipforge.native_window"; }
            }
            else
            {
                var w = GuessPythonw(pythonOrPyw);
                if (!string.IsNullOrWhiteSpace(w) && File.Exists(w)) exe = w;
            }
        }
        catch { }

        var psi = new ProcessStartInfo(exe, args)
        {
            WorkingDirectory = AppDir,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        ApplyChildEnv(psi);
        var p = Process.Start(psi);
        Log("Native window started PID=" + (p != null ? p.Id.ToString() : "?") + " exe=" + exe);
    }

    static void ApplyChildEnv(ProcessStartInfo psi)
    {
        try
        {
            var bin = Path.Combine(AppDir, "bin");
            var path = psi.EnvironmentVariables.ContainsKey("PATH") ? psi.EnvironmentVariables["PATH"] : Environment.GetEnvironmentVariable("PATH");
            if (Directory.Exists(bin))
                path = bin + Path.PathSeparator + (path ?? "");
            psi.EnvironmentVariables["PATH"] = path ?? "";
        }
        catch { }
    }

    static bool IsServerRunning()
    {
        try
        {
            using (var client = new HttpClient())
            {
                client.Timeout = TimeSpan.FromSeconds(2);
                var res = client.GetAsync(HealthCheckUrl).Result;
                return res.IsSuccessStatusCode;
            }
        }
        catch { return false; }
    }

    static bool WaitForServerReady()
    {
        for (int i = 1; i <= 90; i++) // allow time for first-run pip install/start
        {
            if (IsServerRunning()) { Log("Server ready after " + i + " checks"); return true; }
            Thread.Sleep(1000);
        }
        return false;
    }
}
