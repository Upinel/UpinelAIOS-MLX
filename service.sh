#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Nova Upinel Chow, MSc, LLM, BBA, MENSA  ·  upinel@me.com  ·  upinel.com
#  Copyright (c) 2026 Nova Upinel Chow. All rights reserved.
#
#  Upinel Personal Free License: free for personal use, commercial use by
#  written permission, and anything built from this must credit the author.
#  See LICENSE.
#
#  "Make it work, make it right, make it fast - then measure it, because
#   the third one is only a claim until the numbers agree."
# ─────────────────────────────────────────────────────────────────────────────
# Optional: run the endpoint automatically at login via launchd.
#
#   ./service.sh install     install and start a login service
#   ./service.sh uninstall   stop and remove it
#   ./service.sh status      show what launchd thinks
#
# Not installed by default, and deliberately so: an inference server that
# starts itself holds tens of GB of unified memory whether or not you are using
# it. Install this only on a Mac you are happy to leave serving.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

LABEL="com.upinel.upinelaios"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

case "${1:-}" in
  install)
    step "Installing the login service"

    [[ -d "$MODEL_DIR" ]] || die "Model not found. Run ./install.sh first."

    mkdir -p "$HOME/Library/LaunchAgents" "$RUN_DIR"

    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>${REPO_DIR}/start.sh</string>
    <string>--foreground</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${REPO_DIR}</string>

  <!-- Start at login and keep it alive. ThrottleInterval stops a crash loop
       from spinning: if the model cannot load, launchd waits 120s between
       attempts instead of retrying forever. -->
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>120</integer>

  <key>StandardOutPath</key>
  <string>${RUN_DIR}/launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>${RUN_DIR}/launchd.err.log</string>

  <!-- The process needs the same PATH a login shell has, because mtplx lives
       in /opt/homebrew/bin. -->
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key>
    <string>${HOME}</string>
  </dict>
</dict>
</plist>
EOF

    # Unload any previous copy first, or launchctl errors out.
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST" \
      || die "launchctl bootstrap failed. Inspect: launchctl print gui/$(id -u)/${LABEL}"

    ok "Installed. It will start now, and again at every login."
    log "    Check:  ./service.sh status"
    log "    Remove: ./service.sh uninstall"
    ;;

  uninstall)
    step "Removing the login service"
    if [[ -f "$PLIST" ]]; then
      launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
      rm -f "$PLIST"
      ok "Removed $PLIST"
    else
      info "No service installed."
    fi
    # KeepAlive would restart it otherwise, so make sure nothing is left.
    "$REPO_DIR/stop.sh" 2>/dev/null || true
    ;;

  status)
    step "Login service status"
    if [[ -f "$PLIST" ]]; then
      ok "Installed at $PLIST"
      launchctl print "gui/$(id -u)/${LABEL}" 2>/dev/null \
        | grep -E '^\s+(state|pid|last exit code|program) ' || true
    else
      info "Not installed."
    fi
    log ""
    if server_healthy; then
      ok "Endpoint is answering on port $PORT."
    else
      warn "Endpoint is not answering on port $PORT."
    fi
    ;;

  *)
    show_usage "$0"
    exit 1
    ;;
esac
