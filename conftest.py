# Root conftest: puts the repo root on sys.path so tests can `import Ex0`.


def read_frames(path):
    """Read every frame back out of a written mp4 (acceptance probes)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames
