import pytube


def download_video(url, output_path):
    print("Downloading video...")
    yt = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
    yt = yt.streams.get_highest_resolution().download(output_path)
    return yt
