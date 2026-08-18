//! flwr — a gamified terminal aesthetic for the hos engine.
//!
//! Color bars, framed panels, a growth "organism", and live status — so training,
//! chat, and (later) ripple can be *watched*, not narrated. Zero deps: raw ANSI
//! truecolor. Shared by the `flwr` front-end and `hos` training.

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";

/// True when the terminal has a light background, so the palette flips to
/// dark-on-light (readable black-ish text and saturated accents on white).
/// Decided once from `FLWR_THEME` (light|dark) or the `COLORFGBG` hint most
/// terminals export; defaults to the dark palette when unknown.
fn light_mode() -> bool {
    use std::sync::OnceLock;
    static LIGHT: OnceLock<bool> = OnceLock::new();
    *LIGHT.get_or_init(|| {
        match std::env::var("FLWR_THEME").map(|v| v.to_lowercase()).as_deref() {
            Ok("light") => return true,
            Ok("dark") => return false,
            _ => {}
        }
        if let Ok(v) = std::env::var("COLORFGBG") {
            if let Some(bg) = v.rsplit(';').next() {
                if let Ok(n) = bg.trim().parse::<i32>() {
                    // 7 = light gray, 15 = bright white, 9..=15 = bright colors
                    return n == 7 || (9..=15).contains(&n);
                }
            }
        }
        false
    })
}

fn no_color() -> bool {
    use std::sync::OnceLock;
    static NC: OnceLock<bool> = OnceLock::new();
    *NC.get_or_init(|| std::env::var_os("NO_COLOR").is_some())
}

/// Pick the dark-bg or light-bg color; empty when NO_COLOR is set.
fn fg(dark: (u8, u8, u8), light: (u8, u8, u8)) -> String {
    if no_color() {
        return String::new();
    }
    let (r, g, b) = if light_mode() { light } else { dark };
    format!("\x1b[38;2;{r};{g};{b}m")
}

// ---- palette: muted botanical. Each color carries a dark-bg pastel and a
// deeper light-bg value so the same names read well on white or black.
pub fn petal() -> String {
    fg((206, 142, 168), (168, 58, 104))
} // flwr brand rose (= the wordmark)
pub fn deep() -> String {
    fg((176, 110, 140), (146, 52, 92))
} // darker rose (petal body, same family)
pub fn pollen() -> String {
    fg((214, 186, 134), (150, 104, 24))
} // soft cream center
pub fn stem() -> String {
    fg((138, 162, 128), (74, 108, 60))
} // sage
pub fn root() -> String {
    fg((96, 120, 102), (60, 88, 66))
} // moss
pub fn signal() -> String {
    fg((126, 172, 182), (34, 108, 124))
} // muted teal
pub fn dream() -> String {
    fg((158, 146, 188), (96, 76, 150))
} // soft lavender
pub fn mem() -> String {
    fg((194, 150, 170), (150, 74, 108))
} // dusty rose
pub fn ctx() -> String {
    fg((134, 172, 162), (48, 108, 94))
} // muted sage-teal
pub fn slate() -> String {
    fg((104, 108, 126), (150, 152, 164))
} // frame
pub fn ink() -> String {
    fg((206, 208, 218), (28, 30, 38))
} // body text: light gray on dark, near-black on light
pub fn faint() -> String {
    fg((108, 113, 130), (128, 130, 144))
}

/// Visible width of a string, ignoring ANSI escape sequences.
pub fn vis_len(s: &str) -> usize {
    let mut n = 0;
    let mut esc = false;
    for ch in s.chars() {
        if esc {
            if ch == 'm' {
                esc = false;
            }
            continue;
        }
        if ch == '\x1b' {
            esc = true;
            continue;
        }
        n += 1;
    }
    n
}

fn center(s: &str, w: usize) -> String {
    let vl = vis_len(s);
    if vl >= w {
        return s.to_string();
    }
    let pad = w - vl;
    let lp = pad / 2;
    format!("{}{}{}", " ".repeat(lp), s, " ".repeat(pad - lp))
}

/// A labelled progress bar: `LABEL  ████████░░  81%`.
pub fn bar(label: &str, frac: f32, color: &str) -> String {
    let w = 12usize;
    let frac = frac.clamp(0.0, 1.0);
    let fill = (frac * w as f32).round() as usize;
    let pct = (frac * 100.0).round() as i32;
    format!(
        "{}{}{:<12}{} {color}{}{}{}{} {}{:>3}%{}",
        BOLD,
        faint(),
        label,
        RESET,
        "█".repeat(fill),
        faint(),
        "░".repeat(w - fill),
        RESET,
        ink(),
        pct,
        RESET
    )
}

/// The growth organism at a given `stage` in 0..1 (seed → sprout → bloom).
pub fn organism(stage: f32) -> Vec<String> {
    let (p, d, c, s, r) = (petal(), deep(), pollen(), stem(), root());
    let raw: Vec<String> = if stage < 0.34 {
        vec![
            format!("{d}··{RESET}"),
            format!("{d}░░ {c}▓{RESET} {d}░░{RESET}"),
            format!("{s}█{RESET}"),
            format!("{r}╱ █ ╲{RESET}"),
        ]
    } else if stage < 0.67 {
        vec![
            format!("{d}░░{RESET}"),
            format!("{p}░░ {c}▓▓▓{RESET} {p}░░{RESET}"),
            format!("{d}░░{RESET}"),
            format!("{s}██{RESET}"),
            format!("{r}██ ██ ██{RESET}"),
            format!("{r}███   ███{RESET}"),
        ]
    } else {
        // full bloom: a clean cross-flower (matches the FLWR OS pixel flower) —
        // dithered tips, a ring of petals, a warm pollen center, stem + roots.
        vec![
            format!("{d}░░{RESET}"),
            format!("{p}██ ██ ██{RESET}"),
            format!("{d}░░{RESET} {p}██{RESET} {c}▓▓{RESET} {p}██{RESET} {d}░░{RESET}"),
            format!("{p}██ ██ ██{RESET}"),
            format!("{d}░░{RESET}"),
            format!("{s}██{RESET}"),
            format!("{r}██ ██ ██{RESET}"),
            format!("{r}██   ██   ██{RESET}"),
        ]
    };
    // fixed height (8 rows) so the panel can redraw in place during training
    let mut out: Vec<String> = raw.iter().map(|l| center(l, 30)).collect();
    while out.len() < 8 {
        out.push(center("", 30));
    }
    out.truncate(8);
    out
}

/// A status line: `> MESSAGE`.
pub fn status(msg: &str, color: &str) -> String {
    format!("{}{}>{} {}{}{}", BOLD, color, RESET, color, msg, RESET)
}

/// Frame a set of (already-colored) content lines under a centered title.
pub fn frame(title: &str, lines: &[String], width: usize) -> String {
    // auto-expand so no line overflows the box (visible length, ANSI stripped): a row
    // renders as "║ {line}{pad}║", needing width >= vis_len(line)+1, +1 for a trailing
    // margin. Never shrinks below the requested width.
    let content = lines
        .iter()
        .map(|l| vis_len(l))
        .max()
        .unwrap_or(0)
        .max(vis_len(title));
    let width = width.max(content + 2);
    let fc = slate();
    let mut out = String::new();
    out += &format!("{fc}╔{}╗{RESET}\n", "═".repeat(width));
    let t = format!("{}{}{}{}", BOLD, deep(), title, RESET);
    out += &format!("{fc}║{RESET}{}{fc}║{RESET}\n", center(&t, width));
    out += &format!("{fc}╠{}╣{RESET}\n", "═".repeat(width));
    for l in lines {
        let pad = width.saturating_sub(1 + vis_len(l));
        out += &format!("{fc}║{RESET} {}{}{fc}║{RESET}\n", l, " ".repeat(pad));
    }
    out += &format!("{fc}╚{}╝{RESET}\n", "═".repeat(width));
    out
}

/// The FLWR OS header strip: brand wordmark left, tagline right, a slate rule
/// under it — the terminal echo of the landing page's top chrome.
pub fn os_header() -> String {
    let fc = slate();
    let w = 62usize;
    let left = format!("{BOLD}{}FLWR OS v1.0.0{RESET}", petal());
    let right = format!("{}memory organism · powered by hos{RESET}", faint());
    let gap = w.saturating_sub(vis_len(&left) + vis_len(&right));
    format!(
        "\n  {left}{}{right}\n  {fc}{}{RESET}\n",
        " ".repeat(gap.max(2)),
        "─".repeat(w)
    )
}

/// The bottom nav strip: the FLWR loop + a sign-off, mirroring the ad footer.
pub fn footer_nav() -> String {
    let fc = slate();
    let w = 62usize;
    let left = format!(
        "{BOLD}{}FLWR://{RESET}  {}RESEARCH · BUILD · TRAIN · RUN · DEPLOY · REPEAT{RESET}",
        petal(),
        ink()
    );
    let right = format!("{}stay curious.{RESET} {}█{RESET}", stem(), stem());
    let gap = w.saturating_sub(vis_len(&left) + vis_len(&right));
    format!(
        "  {fc}{}{RESET}\n  {left}{}{right}\n",
        "─".repeat(w),
        " ".repeat(gap.max(2))
    )
}

/// The flwr flower + wordmark: rose petals (= the FLWR text color), green base.
/// Led by the FLWR OS header so the brand moment is the same in app and on-site.
pub fn banner() -> String {
    let b = petal();
    let f = faint();
    let fl = organism(1.0); // full bloom, 8 centered rows
    let mut out = os_header();
    for (i, line) in fl.iter().enumerate() {
        let tag = match i {
            2 => format!("  {BOLD}{b}F L W R{RESET}"),
            4 => format!("  {f}memory organism{RESET}"),
            5 => format!("  {f}powered by hos{RESET}"),
            _ => String::new(),
        };
        out += &format!("  {line}{tag}\n");
    }
    out
}

/// Live training panel: progress + signal bars and the organism at the current
/// growth stage. `signal` and `dream` are 0..1 (e.g. exp(-loss), anneal strength).
pub fn grow_panel(mode: &str, progress: f32, loss: f32, sig: f32, drm: f32) -> String {
    let mut lines = vec![
        bar("MEMORY", progress, &mem()),
        bar("CONTEXT", sig, &ctx()),
        bar("DREAM STATE", drm, &dream()),
        bar("SIGNAL", (1.0 / (1.0 + loss)).clamp(0.0, 1.0), &signal()),
        String::new(),
    ];
    lines.extend(organism(progress));
    frame(&format!("FLWR // {mode}"), &lines, 38)
}
