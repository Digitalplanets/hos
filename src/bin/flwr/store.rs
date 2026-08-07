//! flwr's model store — the provenance-native answer to Ollama's blob registry.
//!
//! Ollama stores models as content-addressed blobs with a text Modelfile bolted
//! on. flwr stores each pulled model in `~/.hos/store/<name>/` next to a
//! `manifest.json` that records, in the spirit of the `.hos` lineage card: where
//! it came from (the `source` + `revision`), a content hash per file, and a
//! combined `identity` hash for the whole artifact. A pulled model is never an
//! anonymous blob — it carries its ancestry.
//!
//! Downloading is the one interop seam, so we shell out to the system `curl`
//! rather than take on an HTTP+TLS dependency: HOS keeps its small,
//! ML-free dependency set, and the network client is the OS's, not the crate's.

use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// One file in a stored model, with its size and content hash.
#[derive(Serialize, Deserialize, Clone)]
pub struct FileEntry {
    pub name: String,
    pub bytes: u64,
    pub hash: String,
}

/// The provenance card for a stored model — manifest.json.
#[derive(Serialize, Deserialize, Clone)]
pub struct Manifest {
    pub name: String,
    pub kind: String,          // "hf" | "gguf"
    pub source: String,        // "hf:<repo>" | "<url>" | "local:<path>"
    pub revision: String,      // hf revision, or "-"
    pub entry: Option<String>, // for gguf: the file to load within the dir
    pub arch: Option<String>,  // read from config.json when available
    pub total_bytes: u64,
    pub pulled_unix: u64,
    pub files: Vec<FileEntry>,
    pub identity: String, // combined content hash — the artifact's lineage id
    /// If this entry was made by `flwr cp`, the store name it descends from.
    /// `#[serde(default)]` so manifests written before this field still load.
    #[serde(default)]
    pub copied_from: Option<String>,
    /// If this entry was produced by `flwr quantize`, the target quant (e.g.
    /// `q8_0`). The source it was derived from is recorded in `copied_from`.
    #[serde(default)]
    pub quant: Option<String>,
}

/// `$FLWR_STORE`, `$HOS_STORE`, else `~/.hos/store`.
pub fn store_root() -> PathBuf {
    if let Ok(d) = std::env::var("FLWR_STORE") {
        return PathBuf::from(d);
    }
    if let Ok(d) = std::env::var("HOS_STORE") {
        return PathBuf::from(d);
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hos/store")
}

/// Resolve a stored model by name to the path the engine should load: the
/// checkpoint directory for an HF pull, or the `.gguf` file for a GGUF pull.
pub fn resolve(name: &str) -> Option<PathBuf> {
    let dir = store_root().join(name);
    let m = read_manifest(&dir)?;
    match m.kind.as_str() {
        "gguf" | "hos" | "registry" => m.entry.map(|e| dir.join(e)),
        _ => Some(dir),
    }
}

/// Read a stored model's manifest by name — for provenance lookups (e.g. saved
/// chat transcripts recording which model produced them).
pub fn lookup(name: &str) -> Option<Manifest> {
    read_manifest(&store_root().join(name))
}

fn read_manifest(dir: &Path) -> Option<Manifest> {
    let bytes = std::fs::read(dir.join("manifest.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn write_manifest(dir: &Path, m: &Manifest) {
    if let Ok(s) = serde_json::to_string_pretty(m) {
        let _ = std::fs::write(dir.join("manifest.json"), s);
    }
}

/// Every stored model's manifest, sorted by name.
pub fn all() -> Vec<Manifest> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(store_root()) {
        for e in rd.flatten() {
            if let Some(m) = read_manifest(&e.path()) {
                out.push(m);
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

// ---- commands --------------------------------------------------------------

/// `flwr pull <spec>` — a HuggingFace repo id (`Qwen/Qwen2.5-0.5B-Instruct`) or
/// a direct `.gguf` URL. Downloads into the store and writes the manifest.
pub fn pull(spec: &str, revision: &str, name_override: Option<&str>) {
    if spec.ends_with(".gguf")
        || spec.ends_with(".hos")
        || spec.starts_with("http://")
        || spec.starts_with("https://")
    {
        pull_gguf_url(spec, name_override);
    } else if !spec.contains('/') {
        // A bare name resolves against the flwr hub. Defaults to the flwr model
        // registry (https://www.flwr.systems) so `flwr pull flwr-bloom` works
        // out of the box; set FLWR_REGISTRY to point at your own hub instead.
        let reg = std::env::var("FLWR_REGISTRY")
            .ok()
            .filter(|r| !r.trim().is_empty())
            .unwrap_or_else(|| "https://www.flwr.systems".to_string());
        pull_registry(reg.trim().trim_end_matches('/'), spec, name_override);
    } else {
        pull_hf(spec, revision, name_override);
    }
}

fn pull_hf(repo: &str, revision: &str, name_override: Option<&str>) {
    let name = name_override
        .map(String::from)
        .unwrap_or_else(|| repo.rsplit('/').next().unwrap_or(repo).to_string());
    let dir = store_root().join(&name);
    eprintln!("  pulling hf:{repo}@{revision}");
    eprintln!("    into {}", dir.display());

    let api = format!("https://huggingface.co/api/models/{repo}/revision/{revision}");
    let meta = curl_string(&api).unwrap_or_else(|e| fail(&format!("HF API: {e}")));
    let v: serde_json::Value =
        serde_json::from_str(&meta).unwrap_or_else(|e| fail(&format!("HF API JSON: {e}")));
    let files: Vec<String> = v["siblings"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s["rfilename"].as_str().map(String::from))
                .filter(|f| wanted(f))
                .collect()
        })
        .unwrap_or_default();
    if !files.iter().any(|f| f.ends_with(".safetensors")) {
        fail("no .safetensors in that repo — HOS loads safetensors, not .bin");
    }

    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&format!("mkdir: {e}")));
    let mut entries = Vec::new();
    let mut total = 0u64;
    for f in &files {
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{f}");
        let dest = dir.join(f);
        eprintln!("    · {f}");
        curl_file(&url, &dest).unwrap_or_else(|e| fail(&format!("download {f}: {e}")));
        let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        total += bytes;
        entries.push(FileEntry {
            name: f.clone(),
            bytes,
            hash: fnv_file(&dest).unwrap_or_default(),
        });
    }

    let m = Manifest {
        name: name.clone(),
        kind: "hf".into(),
        source: format!("hf:{repo}"),
        revision: revision.into(),
        entry: None,
        arch: read_arch(&dir),
        total_bytes: total,
        pulled_unix: now_unix(),
        identity: identity_of(&entries),
        files: entries,
        copied_from: None,
        quant: None,
    };
    write_manifest(&dir, &m);
    eprintln!("  ✓ {name}  ({})  identity {}", human(total), &m.identity);
    eprintln!("    lineage: hf:{repo}@{revision}");
}

fn pull_gguf_url(url: &str, name_override: Option<&str>) {
    let file = url.rsplit('/').next().unwrap_or("model.gguf").to_string();
    let name = name_override
        .map(String::from)
        .unwrap_or_else(|| file.trim_end_matches(".gguf").to_string());
    let dir = store_root().join(&name);
    eprintln!("  pulling {url}");
    eprintln!("    into {}", dir.display());
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&format!("mkdir: {e}")));
    let dest = dir.join(&file);
    eprintln!("    · {file}");
    curl_file(url, &dest).unwrap_or_else(|e| fail(&format!("download: {e}")));
    let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    let entries = vec![FileEntry {
        name: file.clone(),
        bytes,
        hash: fnv_file(&dest).unwrap_or_default(),
    }];
    let m = Manifest {
        name: name.clone(),
        kind: "gguf".into(),
        source: url.into(),
        revision: "-".into(),
        entry: Some(file),
        arch: None,
        total_bytes: bytes,
        pulled_unix: now_unix(),
        identity: identity_of(&entries),
        files: entries,
        copied_from: None,
        quant: None,
    };
    write_manifest(&dir, &m);
    eprintln!("  ✓ {name}  ({})  identity {}", human(bytes), &m.identity);
}

/// `flwr pull <name>` against a self-hosted hub (FLWR_REGISTRY). Resolves the
/// name via `GET {base}/v1/pull/<name>` (a small JSON: url + sha256 + provenance),
/// downloads the `.hos` capsule straight from wherever the hub points (e.g. R2 —
/// the download bytes never touch the hub), and **verifies the advertised
/// sha256** before keeping it. This is how a model descends from your family
/// rather than an anonymous blob: the manifest records the hub + source lineage.
fn pull_registry(base: &str, name: &str, name_override: Option<&str>) {
    let api = format!("{base}/v1/pull/{name}");
    eprintln!("  resolving {name} from {base}");
    let meta = curl_string(&api).unwrap_or_else(|e| fail(&format!("registry {base}: {e}")));
    let v: serde_json::Value =
        serde_json::from_str(&meta).unwrap_or_else(|e| fail(&format!("registry JSON: {e}")));
    let dl_url = v["url"]
        .as_str()
        .unwrap_or_else(|| fail(&format!("registry response for '{name}' has no 'url'")));
    let want_sha = v["sha256"].as_str().unwrap_or("");

    let store_name = name_override.map(String::from).unwrap_or_else(|| name.to_string());
    let file = format!("{store_name}.hos");
    let dir = store_root().join(&store_name);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&format!("mkdir: {e}")));
    let dest = dir.join(&file);
    eprintln!("    · {file}");
    curl_file(dl_url, &dest).unwrap_or_else(|e| fail(&format!("download: {e}")));

    // Integrity gate: the whole point of a provenance-native hub. A mismatch
    // means a corrupted or tampered download — refuse to keep it.
    if !want_sha.is_empty() {
        match sha256_file(&dest) {
            Some(got) if got.eq_ignore_ascii_case(want_sha) => eprintln!("    ✓ sha256 verified"),
            Some(got) => {
                let _ = std::fs::remove_dir_all(&dir);
                fail(&format!("sha256 mismatch (hub {want_sha}, got {got}) — refused"));
            }
            None => eprintln!("    ! no sha256 tool found — kept UNVERIFIED"),
        }
    }

    let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    let entries = vec![FileEntry {
        name: file.clone(),
        bytes,
        hash: fnv_file(&dest).unwrap_or_default(),
    }];
    let m = Manifest {
        name: store_name.clone(),
        kind: "registry".into(),
        source: base.to_string(),
        revision: "-".into(),
        entry: Some(file),
        arch: v["arch"].as_str().map(String::from),
        total_bytes: bytes,
        pulled_unix: now_unix(),
        identity: identity_of(&entries),
        files: entries,
        copied_from: None,
        quant: v["quant"].as_str().map(String::from),
    };
    write_manifest(&dir, &m);
    eprintln!("  ✓ {store_name}  ({})  from {base}", human(bytes));
}

/// sha256 of a file via the system tool — an interop seam like `curl`, so the
/// engine keeps its no-crypto-dependency stance. Prefers `shasum` (macOS/BSD),
/// falls back to `sha256sum` (Linux). Returns the lowercase hex digest.
fn sha256_file(path: &Path) -> Option<String> {
    let attempts: [(&str, &[&str]); 2] = [("shasum", &["-a", "256"]), ("sha256sum", &[])];
    for (bin, args) in attempts {
        if let Ok(out) = Command::new(bin).args(args).arg(path).output() {
            if out.status.success() {
                if let Some(hex) = String::from_utf8_lossy(&out.stdout).split_whitespace().next() {
                    return Some(hex.to_lowercase());
                }
            }
        }
    }
    None
}

/// `flwr list` — the stored models, with provenance.
pub fn list() {
    let models = all();
    if models.is_empty() {
        println!("  no models in the store ({})", store_root().display());
        println!("  pull one:  flwr pull Qwen/Qwen2.5-0.5B-Instruct");
        return;
    }
    println!("  store: {}", store_root().display());
    let (name, size, arch, source) = ("NAME", "SIZE", "ARCH", "SOURCE");
    println!("  {name:<32} {size:>8}  {arch:<7} {source}");
    for m in models {
        println!(
            "  {:<32} {:>8}  {:<7} {}",
            truncate(&m.name, 32),
            human(m.total_bytes),
            m.arch.clone().unwrap_or_else(|| "-".into()),
            m.source
        );
    }
}

/// `flwr show <name>` — the full provenance card.
pub fn show(name: &str) {
    let dir = store_root().join(name);
    let Some(m) = read_manifest(&dir) else {
        fail(&format!("no stored model named '{name}' (try: flwr list)"));
    };
    println!("  +------------------------------------------------------------+");
    println!("  |  flwr show  ·  provenance card                             |");
    println!("  +------------------------------------------------------------+");
    println!("    name      {}", m.name);
    println!("    kind      {}", m.kind);
    println!("    source    {}", m.source);
    println!("    revision  {}", m.revision);
    println!("    arch      {}", m.arch.unwrap_or_else(|| "-".into()));
    println!("    size      {}", human(m.total_bytes));
    println!("    pulled    {} (unix)", m.pulled_unix);
    println!("    identity  {}", m.identity);
    if let Some(q) = &m.quant {
        let from = m.copied_from.as_deref().unwrap_or("source");
        println!("    quant     {q}  (derived from {from})");
    } else if let Some(parent) = &m.copied_from {
        println!("    copy of   {parent}  (same content identity)");
    }
    println!("    path      {}", dir.display());
    println!("    files:");
    for f in &m.files {
        println!(
            "      {:<40} {:>8}  {}",
            truncate(&f.name, 40),
            human(f.bytes),
            f.hash
        );
    }
}

/// `flwr rm <name>` — delete a stored model and its manifest.
pub fn rm(name: &str) {
    let dir = store_root().join(name);
    let Some(m) = read_manifest(&dir) else {
        die(
            "rm",
            &format!("no stored model named '{name}' (try: flwr list)"),
        );
    };
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        die("rm", &format!("could not remove {}: {e}", dir.display()));
    }
    println!("  ✓ removed {name}  ({} freed)", human(m.total_bytes));
}

/// `flwr cp <src> <dst>` — put a model into the store under name `<dst>`.
///
/// `<src>` is either an existing store entry (duplicated, keeping the same
/// content `identity` plus a `copied_from` lineage edge) or a filesystem path
/// (an HF checkpoint directory or a `.gguf` file) imported with
/// `source: local:<abspath>`. Either way the destination must not exist.
pub fn cp(src: &str, dst: &str) {
    let dst_dir = store_root().join(dst);
    if dst_dir.exists() {
        die(
            "cp",
            &format!("'{dst}' already exists — pick another name or `flwr rm {dst}` first"),
        );
    }

    // 1) a bare name that's an existing store entry → duplicate with a lineage
    //    edge. Gated on "no path separator" so a path is never mistaken for a
    //    store name (and vice versa).
    if !src.contains('/') {
        let src_dir = store_root().join(src);
        if let Some(mut m) = read_manifest(&src_dir) {
            copy_files(&src_dir, &dst_dir, &m.files);
            m.name = dst.to_string();
            m.copied_from = Some(src.to_string());
            m.pulled_unix = now_unix();
            write_manifest(&dst_dir, &m);
            println!("  ✓ copied {src} -> {dst}  ({})", human(m.total_bytes));
            println!("    lineage: copy of {src}  (identity {})", m.identity);
            return;
        }
    }

    // 2) a filesystem path → import it into the store.
    let p = Path::new(src);
    if p.exists() {
        import_path(p, dst, &dst_dir);
        return;
    }

    die(
        "cp",
        &format!("'{src}' is neither a stored model (flwr list) nor an existing path"),
    );
}

/// Import an on-disk HF checkpoint directory or `.gguf` file into the store.
fn import_path(src: &Path, dst: &str, dst_dir: &Path) {
    let abs = std::fs::canonicalize(src).unwrap_or_else(|_| src.to_path_buf());
    std::fs::create_dir_all(dst_dir).unwrap_or_else(|e| die("cp", &format!("mkdir: {e}")));

    // Each source is (path on disk, file name within the store dir).
    let (kind, entry, sources): (String, Option<String>, Vec<(PathBuf, String)>) = if src.is_dir() {
        // HF checkpoint: import exactly the files HOS loads.
        let mut srcs = Vec::new();
        if let Ok(rd) = std::fs::read_dir(src) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if wanted(&n) {
                    srcs.push((src.join(&n), n));
                }
            }
        }
        if !srcs.iter().any(|(_, n)| n.ends_with(".safetensors")) {
            let _ = std::fs::remove_dir_all(dst_dir);
            die(
                "cp",
                "that directory has no .safetensors — not an HF checkpoint HOS can load",
            );
        }
        ("hf".to_string(), None, srcs)
    } else if src.extension().and_then(|e| e.to_str()) == Some("gguf") {
        let file = src
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model.gguf".into());
        (
            "gguf".to_string(),
            Some(file.clone()),
            vec![(src.to_path_buf(), file)],
        )
    } else if src.extension().and_then(|e| e.to_str()) == Some("hos") {
        let file = src
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model.hos".into());
        (
            "hos".to_string(),
            Some(file.clone()),
            vec![(src.to_path_buf(), file)],
        )
    } else {
        let _ = std::fs::remove_dir_all(dst_dir);
        die(
            "cp",
            "path must be an HF checkpoint directory, a .gguf, or a .hos file",
        );
    };

    let mut entries = Vec::new();
    let mut total = 0u64;
    for (from, name) in &sources {
        let to = dst_dir.join(name);
        if let Err(e) = std::fs::copy(from, &to) {
            let _ = std::fs::remove_dir_all(dst_dir);
            die("cp", &format!("copy {name}: {e}"));
        }
        let bytes = std::fs::metadata(&to).map(|m| m.len()).unwrap_or(0);
        total += bytes;
        entries.push(FileEntry {
            name: name.clone(),
            bytes,
            hash: fnv_file(&to).unwrap_or_default(),
        });
    }

    let m = Manifest {
        name: dst.to_string(),
        kind,
        source: format!("local:{}", abs.display()),
        revision: "-".into(),
        entry,
        arch: read_arch(dst_dir),
        total_bytes: total,
        pulled_unix: now_unix(),
        identity: identity_of(&entries),
        files: entries,
        copied_from: None,
        quant: None,
    };
    write_manifest(dst_dir, &m);
    println!(
        "  ✓ imported {} -> {dst}  ({})",
        abs.display(),
        human(total)
    );
    println!(
        "    lineage: local:{}  (identity {})",
        abs.display(),
        m.identity
    );
}

/// Copy a set of (flat) files between two store directories, cleaning up a
/// partial copy on failure.
fn copy_files(src_dir: &Path, dst_dir: &Path, files: &[FileEntry]) {
    std::fs::create_dir_all(dst_dir).unwrap_or_else(|e| die("cp", &format!("mkdir: {e}")));
    for f in files {
        if let Err(e) = std::fs::copy(src_dir.join(&f.name), dst_dir.join(&f.name)) {
            let _ = std::fs::remove_dir_all(dst_dir);
            die("cp", &format!("copy {}: {e}", f.name));
        }
    }
}

/// `flwr quantize <src> <dst> [--type q8_0]` — produce a smaller derived GGUF
/// from a GGUF source, recording a lineage edge to the parent. GGUF→GGUF only
/// today; HF→GGUF quantization (which must synthesize tokenizer metadata) is a
/// later milestone.
pub fn quantize(src: &str, dst: &str, target: &str, awq: bool) {
    let Some(ggml) = hos::gguf_write::target_type(target) else {
        die(
            "quantize",
            &format!("unsupported target '{target}' (q8_0 | q4_0 | q5_0 | q4_k | q5_k | q6_k)"),
        );
    };
    let dst_dir = store_root().join(dst);
    if dst_dir.exists() {
        die(
            "quantize",
            &format!("'{dst}' already exists — pick another name or `flwr rm {dst}` first"),
        );
    }

    // Resolve the source to a .gguf file (a store name or a path).
    let src_path = if Path::new(src).exists() && !Path::new(src).is_dir() {
        PathBuf::from(src)
    } else if let Some(p) = resolve(src) {
        p
    } else {
        die(
            "quantize",
            &format!("no source '{src}' (a stored model name or a .gguf path)"),
        );
    };
    if src_path.is_dir() || src_path.extension().and_then(|e| e.to_str()) != Some("gguf") {
        die(
            "quantize",
            "source must be a GGUF model — HF→GGUF quantization isn't supported yet \
             (pull/convert to GGUF first)",
        );
    }

    let g = match hos::gguf::Gguf::open(&src_path) {
        Ok(g) => g,
        Err(e) => die("quantize", &format!("open {}: {e}", src_path.display())),
    };
    let arch = g.meta_str("general.architecture").map(String::from);

    std::fs::create_dir_all(&dst_dir).unwrap_or_else(|e| die("quantize", &format!("mkdir: {e}")));

    // Activation-aware path: calibrate, scale salient channels, write a runnable
    // .hos capsule. Beats plain RTN at low bit-widths (see HOS quant whitepaper).
    if awq {
        let out_file = format!("{dst}.hos");
        let out_path = dst_dir.join(&out_file);
        let src_bytes = std::fs::metadata(&src_path).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "  awq-quantizing {} -> {dst} ({target}) — calibrating ...",
            src_path.display()
        );
        let touched = match hos::hos_awq::awq_quantize_gguf(
            &src_path,
            &out_path,
            ggml,
            0.5,
            hos::hos_awq::CALIB_CORPUS,
        ) {
            Ok(t) => t,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dst_dir);
                die("quantize", &format!("awq: {e}"));
            }
        };
        let out_bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
        let entries = vec![FileEntry {
            name: out_file.clone(),
            bytes: out_bytes,
            hash: fnv_file(&out_path).unwrap_or_default(),
        }];
        let parent = if !src.contains('/') && read_manifest(&store_root().join(src)).is_some() {
            Some(src.to_string())
        } else {
            None
        };
        let m = Manifest {
            name: dst.to_string(),
            kind: "hos".into(),
            source: format!("awq:{src} ({target})"),
            revision: "-".into(),
            entry: Some(out_file),
            arch,
            total_bytes: out_bytes,
            pulled_unix: now_unix(),
            identity: identity_of(&entries),
            files: entries,
            copied_from: parent,
            quant: Some(format!("{target}+awq")),
        };
        write_manifest(&dst_dir, &m);
        let pct = 100.0 * out_bytes as f64 / src_bytes.max(1) as f64;
        println!(
            "  ✓ awq-quantized {src} -> {dst}  ({} -> {}, {pct:.0}% of source; {touched} matrices scaled)",
            human(src_bytes),
            human(out_bytes),
        );
        println!(
            "    lineage: derived from {src} via {target}+awq  (identity {})",
            m.identity
        );
        println!("    verify:  hos --perplexity -m {}", out_path.display());
        return;
    }

    let out_file = format!("{dst}.gguf");
    let out_path = dst_dir.join(&out_file);
    eprintln!("  quantizing {} -> {dst} ({target})", src_path.display());
    let stats = match hos::gguf_write::requantize_gguf(&g, &out_path, ggml) {
        Ok(s) => s,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&dst_dir);
            die("quantize", &format!("{e}"));
        }
    };

    let entries = vec![FileEntry {
        name: out_file.clone(),
        bytes: stats.out_bytes,
        hash: fnv_file(&out_path).unwrap_or_default(),
    }];
    // A store-name parent goes in copied_from (lineage edge); a raw path doesn't.
    let parent = if !src.contains('/') && read_manifest(&store_root().join(src)).is_some() {
        Some(src.to_string())
    } else {
        None
    };
    let m = Manifest {
        name: dst.to_string(),
        kind: "gguf".into(),
        source: format!("derived:{src} ({target})"),
        revision: "-".into(),
        entry: Some(out_file),
        arch,
        total_bytes: stats.out_bytes,
        pulled_unix: now_unix(),
        identity: identity_of(&entries),
        files: entries,
        copied_from: parent,
        quant: Some(target.to_string()),
    };
    write_manifest(&dst_dir, &m);
    let pct = 100.0 * stats.out_bytes as f64 / stats.src_bytes.max(1) as f64;
    println!(
        "  ✓ quantized {src} -> {dst}  ({} -> {}, {pct:.0}% of source; {} of {} tensors)",
        human(stats.src_bytes),
        human(stats.out_bytes),
        stats.quantized,
        stats.tensors
    );
    println!(
        "    lineage: derived from {src} via {target}  (identity {})",
        m.identity
    );
    println!("    verify:  hos --perplexity -m {}", out_path.display());
}

// ---- helpers ---------------------------------------------------------------

/// Exactly the files HOS loads from an HF checkpoint: the weights (including
/// shards and their index), the config, and the tokenizer. A tight whitelist,
/// not "all JSON", so we skip training logs (trainer state, results files),
/// the onnx/coreml subfolders, and the `.bin` duplicates.
fn wanted(f: &str) -> bool {
    if f.contains('/') {
        return false;
    }
    let n = f.to_ascii_lowercase();
    if n.ends_with(".safetensors") || n.ends_with(".index.json") || n == "tokenizer.model" {
        return true;
    }
    matches!(
        n.as_str(),
        "config.json"
            | "generation_config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "special_tokens_map.json"
            | "added_tokens.json"
    )
}

fn read_arch(dir: &Path) -> Option<String> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    v["model_type"]
        .as_str()
        .map(String::from)
        .or_else(|| v["architectures"][0].as_str().map(String::from))
}

fn curl_string(url: &str) -> Result<String, String> {
    let out = Command::new("curl")
        .args(["-fsSL", url])
        .output()
        .map_err(|e| format!("curl not available: {e}"))?;
    if !out.status.success() {
        return Err(format!("curl exit {:?}", out.status.code()));
    }
    String::from_utf8(out.stdout).map_err(|e| e.to_string())
}

fn curl_file(url: &str, dest: &Path) -> Result<(), String> {
    let status = Command::new("curl")
        .args(["-fL", "--progress-bar", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .map_err(|e| format!("curl not available: {e}"))?;
    if !status.success() {
        return Err(format!("curl exit {:?}", status.code()));
    }
    Ok(())
}

/// FNV-1a over a file's bytes — the same hash family `.hos` lineage uses.
fn fnv_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

/// A combined identity for the whole artifact: FNV over "name:hash" lines.
fn identity_of(entries: &[FileEntry]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for e in entries {
        for b in format!("{}:{}\n", e.name, e.hash).bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn human(bytes: u64) -> String {
    const K: f64 = 1024.0;
    let b = bytes as f64;
    if b >= K * K * K {
        format!("{:.1}G", b / (K * K * K))
    } else if b >= K * K {
        format!("{:.1}M", b / (K * K))
    } else if b >= K {
        format!("{:.1}K", b / K)
    } else {
        format!("{bytes}B")
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n.saturating_sub(1)])
    }
}

fn fail(msg: &str) -> ! {
    die("pull", msg)
}

fn die(ctx: &str, msg: &str) -> ! {
    eprintln!("  ✗ flwr {ctx}: {msg}");
    std::process::exit(1);
}
