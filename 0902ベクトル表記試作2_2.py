import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from pathlib import Path
import os
import re
from matplotlib.patches import FancyArrowPatch
import japanize_matplotlib

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))

def draw_overlay(x, y, sigma, theta, bg_x, bg_y, bg, grp,
                 x_unique, y_unique, dx, dy, cell_len,
                 sigma_col, bg_col, stress_file, sheet_index):

    while True:
        # --- 応力表示範囲 ---
        print("主応力の表示範囲を設定します")
        obs_min = float(np.nanmin(sigma))
        obs_max = float(np.nanmax(sigma))
        print(f"観測値範囲: min={obs_min}, max={obs_max}")
        smin_in = input(f"表示最小値を入力（Enterで {obs_min}）: ").strip()
        smax_in = input(f"表示最大値を入力（Enterで {obs_max}）: ").strip()
        #
        #smax_in = 1.5
        #
        smin = float(smin_in) if smin_in else obs_min
        smax = float(smax_in) if smax_in else obs_max
        if smax <= smin:
            print("最大値は最小値より大きくしてください。終了します。")
            return

        sigma_clipped = np.clip(sigma, smin, smax)
        norm = (sigma_clipped - smin) / (smax - smin)
        half_len = (norm * cell_len) * 0.5

        # --- 背景表示範囲 ---
        print("背景の表示範囲を設定します")
        obs_min_bg = float(np.nanmin(bg))
        obs_max_bg = float(np.nanmax(bg))
        print(f"観測値範囲: min={obs_min_bg}, max={obs_max_bg}")
        back_min = input(f"最小値 vmin を入力（Enterで {obs_min_bg}）: ").strip()
        back_max = input(f"最大値 vmax を入力（Enterで {obs_max_bg}）: ").strip()
        #
        #back_min = -1
        #back_max = 3
        #
        back_min = float(back_min) if back_min else obs_min_bg
        back_max = float(back_max) if back_max else obs_max_bg

        # --- pivot ---
        grid_bg = pd.DataFrame({"x": bg_x, "y": bg_y, "z": bg})
        bg_pivot = grid_bg.pivot_table(index="y", columns="x", values="z")

        grid_grp = pd.DataFrame({"x": bg_x, "y": bg_y, "g": grp})
        grp_pivot = grid_grp.pivot_table(index="y", columns="x", values="g")
        #grp_pivot = grp_pivot.iloc[::-1]   # y方向を反転

        # --- 描画 ---
        fig, ax = plt.subplots(figsize=(15, 12))

        # 背景
        im = ax.imshow(
            bg_pivot.values,
            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
            origin="lower", cmap="jet", vmin=back_min, vmax=back_max
        )
        plt.colorbar(im, ax=ax, label=bg_col)

        # 境界線
        g = grp_pivot.values
        vdiff = g[1:, :] != g[:-1, :]
        hdiff = g[:, 1:] != g[:, :-1]

        Xc, Yc = np.meshgrid(x_unique, y_unique)
        y_edge_v = (Yc[1:, :] + Yc[:-1, :]) * 0.5
        x_left = Xc[:-1, :] - dx * 0.5
        x_right = Xc[:-1, :] + dx * 0.5
        yy, xxL, xxR = y_edge_v[vdiff], x_left[vdiff], x_right[vdiff]
        for yy_i, xl, xr in zip(yy, xxL, xxR):
            ax.plot([xl, xr], [yy_i, yy_i], color="black", linewidth=0.8)

        x_edge_h = (Xc[:, 1:] + Xc[:, :-1]) * 0.5
        y_bottom = Yc[:, :-1] - dy * 0.5
        y_top = Yc[:, :-1] + dy * 0.5
        xx, yyB, yyT = x_edge_h[hdiff], y_bottom[hdiff], y_top[hdiff]
        for xx_i, yb, yt in zip(xx, yyB, yyT):
            ax.plot([xx_i, xx_i], [yb, yt], color="black", linewidth=0.8)

        # 両矢印ベクトル
        
        #ux = half_len * np.cos(theta)
        #uy = half_len * np.sin(theta)
        #for xi, yi, uxi, uyi in zip(x, y, ux, uy):
        #    ax.plot([xi - uxi, xi + uxi], [yi - uyi, yi + uyi],
        #            color="black", linewidth=1.2)
        #plt.show()
        ux = half_len * np.cos(theta)
        uy = half_len * np.sin(theta)
        for xi, yi, uxi, uyi , val in zip(x, y, ux, uy, sigma):
            if val == 0:
                continue  # 値0はスキップ
            ax.plot([xi - uxi, xi + uxi], [yi - uyi, yi + uyi],
                    color="black", linewidth=1.2)
        ax.invert_yaxis()
        plt.show()
        
        
        '''
        ux = half_len * np.cos(theta)
        uy = half_len * np.sin(theta)
        for xi, yi, uxi, uyi in zip(x, y, ux, uy):
            arrow = FancyArrowPatch(
                (xi - uxi, yi - uyi),
                (xi + uxi, yi + uyi),
                arrowstyle='<->',
                mutation_scale=10,
                color='black',
                linewidth=1.2
            )
            ax.add_patch(arrow)
        '''
        '''
        ##ここからが試作
        ux = half_len * np.cos(theta)
        uy = half_len * np.sin(theta)
        for xi, yi, uxi, uyi, val in zip(x, y, ux, uy, sigma):
            if val == 0:
                continue  # 値0はスキップ

        arrow_style = '<->' if val > 0 else '-'
        arrow = FancyArrowPatch(
            (xi - uxi, yi - uyi),
            (xi + uxi, yi + uyi),
            arrowstyle=arrow_style,
            mutation_scale=10,
            color='black',
            linewidth=1.2
        )
        ax.add_patch(arrow)
        ##
'''

        ax.set_aspect("equal")
        ax.set_title(f"Overlay: Stress={sigma_col}, Background={bg_col}")
        plt.show(block=False)

        out_dir = stress_file.parent / "vector_overlay_output"
        out_dir.mkdir(exist_ok=True)
        fname = sanitize_filename(f"{sheet_index}th_{bg_col}_overlay_{smin}_to_{smax}.png")
        save_path = out_dir / fname
        #plt.savefig(save_path, dpi=300, bbox_inches="tight")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"保存しました: {save_path}")

        plt.show(block=False)

        again = input("もう一度範囲を変えて表示しますか？ (y/n): ").strip()
        if again.lower() != "y":
            break

def main():
    root = Tk(); root.withdraw()

    # --- 主応力データのExcel ---
    stress_file = filedialog.askopenfilename(
        title="主応力データのExcelを選択してください",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not stress_file:
        print("主応力ファイル未選択。終了します。")
        return
    stress_file = Path(stress_file)

    nth = int(input("何段階目ですか（0=1枚目のシート）: ").strip())
    sheet_index = nth

    df_stress = pd.read_excel(stress_file, sheet_name=sheet_index)
    df_excel = int(input("背景に使いたい画像が主応力なら1を入力:"))
    #df_excel = 1

    # --- 背景データのExcel ---
    bg_file = filedialog.askopenfilename(
        title="背景用Excelを選択してください",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not bg_file:
        print("背景ファイル未選択。終了します。")
        return
    bg_file = Path(bg_file)
    df_bg = pd.read_excel(bg_file)

    # --- 列名指定 ---
    def get_col(prompt, default, df):
        v = input(f"{prompt}（Enterで '{default}'）: ").strip()
        if v and v not in df.columns:
            print(f"⚠ 指定列 '{v}' が存在しません。デフォルト '{default}' を使用します。")
            return default
        return v if v else default

    x_col = get_col("主応力X列名", "x_position", df_stress)
    y_col = get_col("主応力Y列名", "y_position", df_stress)
    sigma_col = get_col("主応力の大きさ列名", "sigma1", df_stress)
    theta_col = get_col("主応力の角度(度)列名", "theta_p_deg_app", df_stress)

    #bg_x_col = get_col("背景X列名", "x_position", df_bg)
    #bg_y_col = get_col("背景Y列名", "y_position", df_bg)
    #bg_col   = get_col("背景ラスター列名（例: CI）", "GRed_ER_EBSD_σ11", df_bg)

    #とりあえず0903に使用
    '''
    x_col = "X_pixel_"
    y_col = "Y_pixel_"
    sigma_col = "sigma1"
    theta_col = "theta_p_deg_app"

    bg_x_col = "X_pixel_"
    bg_y_col = "Y_pixel_"
    bg_col   = "sigma1"
    '''
    if df_excel == 1:
        bg_x_col = x_col
        bg_y_col = y_col
        bg_col   = get_col("背景ラスター列名（例: CI）", "sigma1", df_stress)
    else:
        bg_x_col = get_col("背景X列名", "x_position", df_bg)
        bg_y_col = get_col("背景Y列名", "y_position", df_bg)
        bg_col   = get_col("背景ラスター列名（例: CI）", "GRed_ER_EBSD_σ11", df_bg)
    
    grp_col  = get_col("境界抽出用グループ列名（例: grain_id）", "c0thEBSDmap_PointTo_Resample2", df_bg)
    #grp_col = "GN_Resample"

    # --- データ取得 ---
    x = df_stress[x_col].to_numpy()
    y = df_stress[y_col].to_numpy()
    sigma = df_stress[sigma_col].to_numpy().astype(float)
    theta_deg = df_stress[theta_col].to_numpy().astype(float)
    theta = np.deg2rad(theta_deg)

    bg_x = df_bg[bg_x_col].to_numpy()
    bg_y = df_bg[bg_y_col].to_numpy()
    if df_excel == 1:
        bg = df_stress[bg_col].to_numpy()
    else:
        bg = df_bg[bg_col].to_numpy()
    bg = df_stress[bg_col].to_numpy()
    grp = df_bg[grp_col].to_numpy()

    # --- 格子情報 ---
    x_unique = np.unique(bg_x); y_unique = np.unique(bg_y)
    x_unique.sort(); y_unique.sort()
    dx = float(np.median(np.diff(x_unique)))
    dy = float(np.median(np.diff(y_unique)))
    cell_len = min(abs(dx), abs(dy))

    # --- 描画関数呼び出し（繰り返し対応）---
    draw_overlay(x, y, sigma, theta, bg_x, bg_y, bg, grp,
                 x_unique, y_unique, dx, dy, cell_len,
                 sigma_col, bg_col, stress_file, sheet_index)

if __name__ == "__main__":
    main()
