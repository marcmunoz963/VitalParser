import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import math
import random
from collections import deque


root = tk.Tk()
root.title("Selección múltiple")
root.geometry("400x500")

# Aixo a posteriori sera analitzat i carregat des del fitxer (variable)
opciones = [
    "ECG", "RR", "ST_I", "ST_II", "ST_III", "NIBP_SYS",
    "ECG", "RR", "ST_I", "ST_II", "ST_III", "NIBP_SYS",
    "ECG", "RR", "ST_I", "ST_II", "ST_III", "NIBP_SYS",
    "ECG", "RR", "ST_I", "ST_II", "ST_III", "NIBP_SYS",
]

# Frame per mantenir Listbox i Scrollbar
main_frame = tk.Frame(root)
main_frame.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)

# Configuracio que fa la finestra expendible (finestra de seleccio inicial)
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Scrollbar i Listbox que s'adaptin a la finestra (configuracio inicial de la scrollbar i listbox)
scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL)
listbox = tk.Listbox(main_frame, selectmode='multiple', yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)
listbox.grid(row=0, column=0, sticky='nsew')
scrollbar.grid(row=0, column=1, sticky='ns')

# Configuracio que fa la listbox expansible
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

# Obrir les opcions de la listbox i omplir la listbox
for opcion in opciones:
    listbox.insert(tk.END, opcion)

# Funcio que s'activa al clicar el boto "Mostrar seleccion"
# Llegeix les opcions seleccionades i obre una finestra nova amb una matriu de graficas
def mostrar_seleccion():
    seleccion = [listbox.get(i) for i in listbox.curselection()]
    if not seleccion: # Si no s'ha seleccionat res
        messagebox.showwarning("Error", "No ha seleccionado nada")
    else: # Si s'ha seleccionat alguna cosa
        n = len(seleccion)
        cols = math.ceil(math.sqrt(n)) # Aquesta es la millor forma que he trobat per repartir les grafiques
        rows = math.ceil(n / cols) # Esta obert a debat si es millor de alguna altre manera pero es perque sigui flexible depenent de les variable seleccionades

        # Comprovar que no existeixi cap altre finestre oberta abans d'obrir-ne una de nova (de resultats) (hi havia un error sino)
        comprovar_finestra(root)

        # Creem la finestra de resultats com a Toplevel perque es pugui tornar a la finestra de seleccio (i diferenciar-la despres per poder-la eliminar en el pas anterior)
        results_root = tk.Toplevel(root)
        # Identificadors per poder eliminar la finestra de resultats si es torna a clicar el boto sense tancar la finestra de resultats (i parar correctament els after jobs)
        root._current_results = results_root
        results_root._jobs = []
        results_root.title("Pagina de Datos para cada dato seleccionado (En graficas)")
        results_root.geometry("800x600")

        # En el moment de clicar que el boto no es pugui clicar fins que no es tanqui la finestra de resultats (per aixi evitar que es pugui obrir 2 finestres a l'hora)
        try:
            btn.config(state='disabled')
        except Exception:
            pass

        # Fer la matriu de cel·les que s'adapti a la finestra
        for r in range(rows):
            results_root.grid_rowconfigure(r, weight=1)
        for c in range(cols):
            results_root.grid_columnconfigure(c, weight=1)

        # Crea el que aniria dins de cada cel·la
        update_jobs = []  # Matriu on es guardaran els after jobs per poder-los eliminar despres (quan es premi volver)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n: # Si ja s'han posat totes les variables seleccionades surt del bucle
                    break # Aixo deixa algun espai en blanc si no es un nombre perfecte (s'hauria de solucionar o si s'implementa algo diferent) (No m'agrada com queda pero no se com fer-ho millor)
                name = seleccion[idx] # Nom de la variable seleccionada
                frame = tk.Frame(results_root, bd=1, relief='solid', padx=6, pady=6) # Frame per cada cel·la (Els espaiats son els que he vist que utilitzen mes)
                frame.grid(row=r, column=c, sticky='nsew', padx=4, pady=4)

                # Label amb el nom de la variable (Ens el podriem saltar pero queda mes clar) (Sino tmb podriem utilizar aixo per fer una descripcio per a nosaltres, tipo ECG = Electrocardiograma, a dalt la que es veu originalment i aqui la descripcio pels no metges)
                lbl = tk.Label(frame, text=name, anchor='w')
                lbl.pack(fill='x')

                # Placeholder per guardar el lloc on hi aniria la grafica, ja he ficat la grafica pero era de abans
                #placeholder = tk.Label(frame, text='[Gráfica reservada]', bg='#eee', fg='#666', bd=1, relief='ridge')
                #placeholder.pack(fill='both', expand=True, pady=(6,0))

                # Canvas on dibuixarem la grafica
                canvas = tk.Canvas(frame, bg='black', bd=1, relief='sunken')
                canvas.pack(fill='both', expand=True, pady=(6,0)) # Un altre cop nose la mitat de les merdes de aqui, fico perque estiguin maques (subjectiu a canvis si voleu)

                # Punts maxims dels grafics (Segurament s'haura de reduir)
                MAX_POINTS = 1000
                # Estat definit pels canvas, per aixi poder anarlo acutallitzant (aixo crea un per a cada grafica)
                state = {'canvas': canvas, 'name': name, 'job': None, 'buffer': deque(maxlen=MAX_POINTS)}

                def draw_random_plot_state(s=state):  # Sincerament ni idea, aixo es Chat Made
                    c = s['canvas']
                    name_local = s['name']
                    c.delete('all')
                    w = c.winfo_width()
                    h = c.winfo_height()
                    if w <= 4 or h <= 4:
                        return
                    pts = list(s['buffer'])
                    if not pts:
                        # nothing to draw yet
                        c.create_text(10, 10, anchor='nw', text=name_local, fill='#333')
                        return
                    # normalize buffer to 0..1
                    minv = min(pts)
                    maxv = max(pts)
                    span = maxv - minv if maxv != minv else 1.0

                    left, right = 8, w - 8
                    top, bottom = 8, h - 8
                    width = right - left
                    height = bottom - top
                    # subtle grid/axes
                    c.create_line(left, bottom, right, bottom, fill='#222')
                    c.create_line(left, top, left, bottom, fill='#222')

                    # compute x positions across the visible buffer
                    count = len(pts)
                    xs = [left + (i / (count-1)) * width for i in range(count)] if count > 1 else [left + width]
                    ys = [top + (1 - ((v - minv) / span)) * height for v in pts]
                    for i in range(len(xs)-1):
                        c.create_line(xs[i], ys[i], xs[i+1], ys[i+1], fill='#1f77b4', width=2)
                    # latest point marker
                    c.create_oval(xs[-1]-2, ys[-1]-2, xs[-1]+2, ys[-1]+2, fill='#1f77b4', outline='')
                    c.create_text(left+6, top+6, anchor='nw', text=name_local, fill='#fff')

                def schedule_update(s=state, interval=500): # Chat Made
                    # only draw/reschedule if both the results window and canvas still exist
                    try:
                        if not results_root.winfo_exists() or not s['canvas'].winfo_exists():
                            return
                    except Exception:
                        return
                    # append one new sample (random here; replace with real data source)
                    try:
                        s['buffer'].append(random.random())
                    except Exception:
                        pass
                    # draw updated buffer
                    draw_random_plot_state(s)
                    # schedule next and save job id (use results_root.after so we can cancel on the same object)
                    try:
                        job = results_root.after(interval, lambda: schedule_update(s, interval))
                        s['job'] = job
                        update_jobs.append(job)
                        try:
                            results_root._jobs.append(job)
                        except Exception:
                            pass
                    except Exception:
                        pass

                # redraw immediately when the canvas is resized
                canvas.bind('<Configure>', lambda e, s=state: draw_random_plot_state(s))
                # start the periodic updates
                schedule_update(state, interval=500)

                idx += 1

        # Oculta la finestra de selecció (root)
        try:
            root.withdraw()
        except Exception:
            pass

        # Boto Volver de la finestra de les Grafiques
        def volver():
            # Acabes amb totes les jobs de les grafiques (Eliminar les grafiques)
            try:
                jobs = getattr(results_root, '_jobs', None)
                if jobs:
                    for job in list(jobs):
                        try:
                            results_root.after_cancel(job)
                        except Exception:
                            pass
                    # Un cop parats i eliminats elimines els _jobs
                    results_root._jobs.clear()
                else: # Per si s'escapa algun que estiguin amb update_jobs (lo mateix)
                    for job in list(update_jobs):
                        try:
                            results_root.after_cancel(job)
                        except Exception:
                            pass
                    update_jobs.clear()
            except Exception:
                pass

            # Tencar la finestra de resultats
            try:
                results_root.destroy()
            except Exception:
                pass

            # Eliminar tots els current_results que quedin
            try:
                if getattr(root, '_current_results', None) is results_root:
                    root._current_results = None
            except Exception:
                pass

            # Fas que el boto anterior torni a funcionar (si, 45 minuts de la meva vida perduda en aixo)
            try:
                btn.config(state='normal')
            except Exception:
                pass

            # Torna a obrir la finestra de seleccio (root)
            try:
                root.deiconify()
            except Exception:
                pass

        # Si tenques la finestra per la 'X' que faci el mateix que el boto de volver
        try:
            results_root.protocol('WM_DELETE_WINDOW', volver)
        except Exception:
            pass

        # Boto de volver
        volver_btn = tk.Button(results_root, text='Volver', command=volver)
        volver_btn.grid(row=rows, column=0, columnspan=cols, sticky='ew', padx=6, pady=6)
        results_root.grid_rowconfigure(rows, weight=0)


def comprovar_finestra(root):
    try: # Buscar que no existeixi ninguna altre finestra oberta (de resultats)
        prev = getattr(root, '_current_results', None)
        if prev is not None and prev.winfo_exists():
            eliminar_finestra(prev) # En cas que si, destrueix la finestra de resultats anterior
    except Exception:
        pass

    # Tambe busca a veure si hi ha alguna finestra oberta (Toplevel) que no s'hagi tancat correctament (tambe pot ser que estiguin en threads diferents)
    try:
        for child in list(root.winfo_children()): # Mira dins dels threads de root 
            try:
                if child.winfo_class() == 'Toplevel' or isinstance(child, tk.Toplevel): # Si algun es Toplevel (llavors es una finestra de resultats)
                    eliminar_finestra(child) # Elimina la finestra
            except Exception:
                pass
    except Exception:
        pass
    
def eliminar_finestra(prev):
    if hasattr(prev, '_jobs'):
        for job in list(prev._jobs): # basicament busca si hi ha algun after job (grafica encara corrent o parada) i l'elimina
            try:
                prev.after_cancel(job)
            except Exception:
                pass
    try:
        prev.destroy() # Destrueix la finestra de resultats anterior
    except Exception:
        pass

# Boto de la finestra original
btn = tk.Button(root, text="Mostrar selección", command=mostrar_seleccion)
btn.grid(row=1, column=0, sticky='ew', padx=8, pady=(0,8))
root.grid_columnconfigure(0, weight=1)

root.mainloop()
