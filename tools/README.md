# Retouch tool (Cakes Terminal house look)

`python3 retouch.py photo1.jpg photo2.jpg ...` -> writes `out/<name>_final.jpg` (1080x1350, 4:5).

Pipeline: even side-to-side lighting -> backdrop color matched to `house_look.json` -> wall/table seam removed
-> cake centered with its board in frame -> backdrop extended where needed -> sRGB JPEG for upload.
The cake itself is never retouched.
