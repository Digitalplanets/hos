# Packaging — `.flwr` / `.hos` file associations

Both extensions are the **same format** (a `.flwr` is a byte-valid `.hos`; the engine
reads by magic bytes, not extension), so both associate to the same handler: `flwr run`.

## Windows
Edit the path in `windows/flwr-assoc.reg` to your `flwr.exe` location (default
`%USERPROFILE%\.cargo\bin\flwr.exe`), then double-click the `.reg` to apply
(per-user, no admin). Double-clicking a `.flwr`/`.hos` then runs it in flwr.

## macOS
Finder file-association needs the type declared by an **app bundle's** Info.plist
(a bare CLI can't own a UTI). Two options:

1. **Quick (no bundle):** use [`duti`](https://github.com/moretension/duti):
   ```bash
   brew install duti
   duti -s <your-flwr-bundle-id> flwr all     # once you ship a .app
   ```
2. **Proper:** ship a thin `flwr.app` wrapper whose `Info.plist` includes the snippet
   in `macos/Info.flwr-uti.plist` — it declares the `.flwr` UTI as **conforming to**
   the `.hos` UTI (so macOS treats `.flwr` as a specialization of the sovereign `.hos`
   type), with `flwr run` as the handler.

The `.plist` snippet is provided as the exported/imported UTI declaration to paste into
that bundle's `Info.plist`.
