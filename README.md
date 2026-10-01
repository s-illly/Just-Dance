# Just Dance

Dance along to any YouTube video with your webcam. You get scored on how closely your moves match the dancers in the video.

## Setup

You need Python 3 and a webcam.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Play

```bash
python main.py
```

Then:

1. Paste a YouTube link to a dance video.
2. Enter how many dancers are in the video (1-4).
3. Enter how many people are playing (1-4).
4. Wait while the video downloads and the dance moves are analysed.
5. Stand back so the webcam can see your whole body, then press **Space** to start.

## Controls

| Key   | Action                              |
|-------|-------------------------------------|
| Space | Start the song / play again at the end |
| Q     | Quit                                |

## Tips

- Use a well-lit room and keep your full body in frame.
- With more than one player, stand side by side, with ample space between players.
