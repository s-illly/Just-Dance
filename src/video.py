import yt_dlp
import tempfile, os 

def download_video(url):
    """
    Download a youtube video as an mp4 into output_dir.
    Returns the path to the saved file
    """
    tmp = tempfile.mktemp(suffix=".mp4")
    base = os.path.splitext(tmp)[0] # strip .mp4

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": base + ".%(ext)s",
        "merge_output_format": "mp4"
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fps = info.get("fps", 30)
        print(f"Downloaded: {info['title']}")
        print(f"Duration:   {info['duration']}s  |  FPS: {fps}")
    
    return base + ".mp4", info
