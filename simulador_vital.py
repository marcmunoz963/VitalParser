import os
import time
import shutil

# -------------------------------
# Configuració
# -------------------------------
src_file = "./base_sample.vital"     # Fitxer original curt (de prova)
dst_file = "./records/prova/stream_sample.vital"   # Fitxer de "streaming"
chunk_size = 1024 * 512                      # 0.5 MB per fragment
interval_secs = 2                            # temps entre escriptures (s)

# -------------------------------
# Inicialització
# -------------------------------
if not os.path.exists(src_file):
    raise FileNotFoundError(f"No s'ha trobat el fitxer base: {src_file}")

os.makedirs(os.path.dirname(dst_file), exist_ok=True)

# Si ja existeix una versió anterior, s'elimina
if os.path.exists(dst_file):
    os.remove(dst_file)
    print(f"🗑️  Fitxer antic eliminat: {dst_file}")

print(f"🚀 Simulant streaming des de {src_file} cap a {dst_file}")
print(f"   Chunk size: {chunk_size/1024:.1f} KB, interval: {interval_secs}s\n")

# -------------------------------
# Bucle de còpia progressiva
# -------------------------------
bytes_total = os.path.getsize(src_file)
bytes_written = 0
start_time = time.time()

with open(src_file, "rb") as fin, open(dst_file, "wb") as fout:
    chunk = fin.read(chunk_size)
    while chunk:
        fout.write(chunk)
        fout.flush()
        bytes_written += len(chunk)

        percent = (bytes_written / bytes_total) * 100
        print(f"[{percent:5.1f}%] {bytes_written/1024/1024:.2f} MB escrits...", end="\r")

        time.sleep(interval_secs)
        chunk = fin.read(chunk_size)

print("\n✅ Simulació completada.")
print(f"Temps total: {time.time() - start_time:.1f}s")

