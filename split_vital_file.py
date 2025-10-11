import os

# -------------------------------
# CONFIGURACIÓ
# -------------------------------
src_file = "./base_sample.vital"      # Fitxer gran d’origen
dst_dir = "./samples/base_samples"    # Carpeta on es guardaran els trossos
chunk_size_mb = 3                     # Mida de cada fragment (en MB)

# -------------------------------
# INICIALITZACIÓ
# -------------------------------
os.makedirs(dst_dir, exist_ok=True)
chunk_size = chunk_size_mb * 1024 * 1024
base_name = os.path.splitext(os.path.basename(src_file))[0]

# -------------------------------
# DIVISIÓ EN FRAGMENTS
# -------------------------------
with open(src_file, "rb") as f:
    idx = 0
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        dst_path = os.path.join(dst_dir, f"{base_name}_part{idx:03d}.vital")
        with open(dst_path, "wb") as out:
            out.write(chunk)
        print(f"🧩 Creat fragment {dst_path} ({len(chunk)/1024/1024:.2f} MB)")
        idx += 1

print(f"\n✅ Divisió completada: {idx} fragments creats a {dst_dir}")
