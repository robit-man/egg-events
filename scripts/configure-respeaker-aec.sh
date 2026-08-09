#!/usr/bin/env bash
set -euo pipefail

raw_source="alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input"
respeaker_sink_prefix="alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00."
respeaker_python="${EGG_RESPEAKER_PYTHON:-python3}"

if ! pactl list sources short | awk '{print $2}' | grep -qx "$raw_source"; then
  echo "configure-respeaker-aec: PulseAudio source '$raw_source' not present yet (USB enumeration or PulseAudio card creation still pending)" >&2
  exit 1
fi
# The card negotiates between an analog-stereo and an iec958-stereo output
# profile across boots/re-enumerations; discover whichever one is actually
# active instead of assuming one, which previously failed silently whenever
# the other profile won.
hardware_sink="$(pactl list sinks short | awk -v prefix="$respeaker_sink_prefix" 'index($2, prefix) == 1 {print $2; exit}')"
if [[ -z "$hardware_sink" ]]; then
  echo "configure-respeaker-aec: no PulseAudio sink with prefix '$respeaker_sink_prefix' present yet (USB enumeration or PulseAudio card creation still pending)" >&2
  exit 1
fi

for module in $(pactl list modules short | awk '$2 == "module-echo-cancel" {print $1}'); do
  pactl unload-module "$module" || true
done

"$respeaker_python" - <<'PY'
import struct
import sys
import usb.core
import usb.util

device = usb.core.find(idVendor=0x2886, idProduct=0x0018)
if device is None:
    print("configure-respeaker-aec: ReSpeaker USB device 2886:0018 not enumerated yet", file=sys.stderr)
    raise SystemExit(1)

parameters = {
    "AGCONOFF": (19, 0, "int", 1),
    "GAMMAVAD_SR": (19, 39, "float", 3.5),
}
for _, (parameter_id, offset, kind, value) in parameters.items():
    payload = struct.pack("iii", offset, int(value), 1) if kind == "int" else struct.pack("ifi", offset, float(value), 0)
    device.ctrl_transfer(
        usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, 0, parameter_id, payload, 100000,
    )
PY

pactl set-default-source "$raw_source"
pactl set-default-sink "$hardware_sink"
