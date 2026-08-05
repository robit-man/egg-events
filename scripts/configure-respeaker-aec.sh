#!/usr/bin/env bash
set -euo pipefail

raw_source="alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input"
hardware_sink="alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo"
respeaker_python="${EGG_RESPEAKER_PYTHON:-python3}"

pactl list sources short | awk '{print $2}' | grep -qx "$raw_source"
pactl list sinks short | awk '{print $2}' | grep -qx "$hardware_sink"

for module in $(pactl list modules short | awk '$2 == "module-echo-cancel" {print $1}'); do
  pactl unload-module "$module" || true
done

"$respeaker_python" - <<'PY'
import struct
import usb.core
import usb.util

device = usb.core.find(idVendor=0x2886, idProduct=0x0018)
if device is None:
    raise SystemExit("ReSpeaker 4 Mic Array is unavailable")

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
