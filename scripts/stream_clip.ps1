<#
.SYNOPSIS
Loops a video clip into MediaMTX so it behaves like an RTSP camera.

.DESCRIPTION
Start MediaMTX first (run `mediamtx` in another terminal), then run this script.
Point config.yaml at it with:  source.type: rtsp, detect_url: rtsp://127.0.0.1:8554/garage

.EXAMPLE
.\scripts\stream_clip.ps1
.\scripts\stream_clip.ps1 -Clip clips\night.mp4
.\scripts\stream_clip.ps1 -Clip clips\iphone.mov -Reencode   # if the stream won't play (HEVC, odd formats)
#>
param(
    [string]$Clip = "clips/test.mp4",
    [string]$Url = "rtsp://127.0.0.1:8554/garage",
    [switch]$Reencode
)

if (-not (Test-Path $Clip)) {
    Write-Error "Clip not found: $Clip"
    exit 1
}

# Copying the video as-is is cheap; re-encoding to H.264 works with any input but uses more CPU.
if ($Reencode) {
    $videoArgs = @("-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", "30")
} else {
    $videoArgs = @("-c:v", "copy")
}

Write-Host "Streaming $Clip to $Url in a loop (Ctrl+C to stop)."
ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i $Clip @videoArgs -an -f rtsp -rtsp_transport tcp $Url
