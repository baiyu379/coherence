import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib as mpl

# STFTを実行する関数
def STFT(t_wndw, n_stft, tm, sig):
    dt = tm[1] - tm[0]

    # 入力された n_wndw を t_wndwで設定した幅より小さい，かつ，2の累乗個に設定する
    n_wndw = int(2**(np.floor(np.log2(t_wndw/dt))))
    t_wndw = n_wndw*dt # recalculate t_wndw
    n_freq = n_wndw

    # 周波数
    freq_sp = np.fft.fftfreq(n_wndw, d=dt)

    # 2Hzから10Hzまでの周波数範囲を抽出
    freq_range_mask = (freq_sp >= 2) & (freq_sp <= 10)
    freq_range_indices = np.where(freq_range_mask)[0]
    freq_range = freq_sp[freq_range_indices]

    # スペクトログラムを計算する時刻を決める
    m = len(tm) - n_wndw
    indxs = np.zeros(n_stft, dtype=int)
    for i in range(n_stft):
        indxs[i] = int(m/(n_stft+1)*(i+1)) + n_wndw//2

    tm_sp = tm[indxs] # DFTをかける時刻の配列

    # スペクトログラムを計算する
    # スペクトルは indxs[i] - n_wndw //2 + 1 ~ indxs[i] + n_wndw//2 の n_wndw 幅で行う
    sp = np.zeros((n_freq, n_stft), dtype=complex) # スペクトログラムの2次元ndarray

    wndw = np.hamming(n_wndw) # hamming窓

    for i in range(n_stft):
        indx = indxs[i] - n_wndw//2 + 1
        sp[:, i] = np.fft.fft(wndw*sig[indx:indx+n_wndw], n_wndw)/np.sqrt(n_wndw)

    # 2Hzから10Hzの周波数範囲のスペクトログラムを取得
    sp_range = sp[freq_range_indices, :]

    return freq_range, tm_sp, sp_range


# 時間または周波数方向に三角窓で平滑化する関数
def smoothing(sp):
    n_freq, n_stft = sp.shape
    sp_smthd = np.zeros_like(sp)

    for i in range(n_stft):
        krnl = np.array([1.0, 2.0, 1.0])
        sp_smthd[:, i] = np.convolve(sp[:, i], krnl, mode='same') / np.sum(krnl)

    for j in range(n_freq):
        krnl = np.array([1.0, 2.0, 1.0])
        sp_smthd[j, :] = np.convolve(sp_smthd[j, :], krnl, mode='same') / np.sum(krnl)

    return sp_smthd


# データの読み込み
data = np.loadtxt('result_file.csv', delimiter=',', skiprows=1)  # ヘッダー行をスキップしてデータを読み込み
time_data = np.loadtxt('time_info.csv', delimiter=',', usecols=(0,), skiprows=1)  # ヘッダー行をスキップして時間情報を読み込み

# 時間情報と信号データの抽出
t = time_data  # 時間情報
# サイドバーでchannel1とchannel2の値を選択
channel1 = st.sidebar.slider('Select channel 1', min_value=0, max_value=33, value=3)
channel2 = st.sidebar.slider('Select channel 2', min_value=0, max_value=33, value=10)

signal1 = data[:, channel1]  # 列1の信号データ
signal2 = data[:, channel2]  # 列2の信号データ
tms = t[0]  # 最初の時間
tme = t[-1]  # 最後の時間

# ウィンドウ幅とSTFTを施す数の設定
t_wndw = 2000.0e-3  # 100 milisecond
n_stft = 200  # number of STFT
freq_upper = 10  # 表示する周波数の上限

# STFTを実行
freq_sp01, tm_sp01, sp01 = STFT(t_wndw, n_stft, t, signal1)
freq_sp02, tm_sp02, sp02 = STFT(t_wndw, n_stft, t, signal2)

# クロススペクト
# クロススペクトル
xsp = sp01 * np.conjugate(sp02)

# コヒーレンスとフェイズ
sp01_pw_smthd = smoothing(np.abs(sp01) ** 2)  # 平滑化
sp02_pw_smthd = smoothing(np.abs(sp02) ** 2)  # 平滑化
xsp_smthd = smoothing(xsp)  # 平滑化

coh = np.abs(xsp_smthd) ** 2 / (sp01_pw_smthd * sp02_pw_smthd)  # （二乗）コヒーレンス
phs = np.rad2deg(np.arctan2(np.imag(xsp_smthd), np.real(xsp_smthd)))  # フェイズ

# 結果のプロット
#------------------------------------------------------------------------------
# 結果のプロット
# 解析結果の可視化
figsize = (210/25.4, 294/25.4)
dpi = 200
fig = plt.figure(figsize=figsize, dpi=dpi)

# 図の設定 (全体)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = 'Arial'

# 窓関数幅をプロット上部に記載
fig.text(0.10, 0.95, f't_wndw = {t_wndw} s')

# プロット枠の設定
ax01 = fig.add_axes([0.125, 0.79, 0.70, 0.08])
ax02 = fig.add_axes([0.125, 0.59, 0.70, 0.08])

ax_sp01 = fig.add_axes([0.125, 0.68, 0.70, 0.10])
cb_sp01 = fig.add_axes([0.85, 0.68, 0.02, 0.10])
ax_sp02 = fig.add_axes([0.125, 0.48, 0.70, 0.10])
cb_sp02 = fig.add_axes([0.85, 0.48, 0.02, 0.10])

ax_xsp = fig.add_axes([0.125, 0.33, 0.70, 0.10])
cb_xsp = fig.add_axes([0.85, 0.33, 0.02, 0.10])
ax_coh = fig.add_axes([0.125, 0.22, 0.70, 0.10])
cb_coh = fig.add_axes([0.85, 0.22, 0.02, 0.10])
ax_phs = fig.add_axes([0.125, 0.10, 0.70, 0.10])
cb_phs = fig.add_axes([0.85, 0.10, 0.02, 0.10])

# ---------------------------
# ---------------------------
# テスト信号 sig01 のプロット
ax01.set_xlim(tms, tme)
ax01.set_xlabel('')
ax01.tick_params(labelbottom=False)
ax01.set_ylabel('x (sig01)')

ax01.plot(t, signal1, c='black')

# ---------------------------
# テスト信号 sig02 のプロット
ax02.set_xlim(tms, tme)
ax02.set_xlabel('')
ax02.tick_params(labelbottom=False)
ax02.set_ylabel('y (sig02)')

ax02.plot(t, signal2, c='black')

# ---------------------------
# テスト信号 sig01 のスペクトログラムのプロット
ax_sp01.set_xlim(tms, tme)
ax_sp01.set_xlabel('')
ax_sp01.tick_params(labelbottom=False)
# スペクトログラムの縦軸の範囲を2Hzからに変更
ax_sp01.set_ylim(2, freq_upper)
ax_sp01.set_ylabel('frequency\n(Hz)')

norm = mpl.colors.Normalize(vmin=np.log10(np.abs(sp01[freq_sp01 < freq_upper, :])**2).min(),
                            vmax=np.log10(np.abs(sp01[freq_sp01 < freq_upper, :])**2).max())
#norm = mpl.colors.Normalize(vmin=-2,vmax=-1)
cmap = mpl.cm.jet

ax_sp01.contourf(tm_sp01, freq_sp01, np.log10(np.abs(sp01)**2),
                 norm=norm, levels=256, cmap=cmap)

ax_sp01.text(0.99, 0.97, "spectrogram", color='white', ha='right', va='top',
              path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                            path_effects.Normal()],
              transform=ax_sp01.transAxes)

##mpl.colorbar.ColorbarBase(cb_sp01, cmap=cmap, norm=norm,
#boundaries=np.linspace(-2, 4, 10),
                          #orientation="vertical",
                          #label='$\log_{10}|X/N|^2$')
mpl.colorbar.ColorbarBase(cb_sp01, cmap=cmap, norm=norm,
                          orientation="vertical",
                          label='$\log_{10}|X/N|^2$',
                          ticks=np.linspace(np.log10(np.abs(sp01[freq_sp01 < freq_upper, :])**2).min(),
                                            np.log10(np.abs(sp01[freq_sp01 < freq_upper, :])**2).max(), 5))

# ---------------------------
# テスト信号 sig02 のスペクトログラムのプロット
ax_sp02.set_xlim(tms, tme)
ax_sp02.set_xlabel('')
ax_sp02.tick_params(labelbottom=True)
# スペクトログラムの縦軸の範囲を2Hzからに変更
ax_sp02.set_ylim(2, freq_upper)
ax_sp02.set_ylabel('frequency\n(Hz)')

norm = mpl.colors.Normalize(vmin=np.log10(np.abs(sp02[freq_sp02 < freq_upper, :])**2).min()+3,
                            vmax=np.log10(np.abs(sp02[freq_sp02 < freq_upper, :])**2).max())

#norm = mpl.colors.Normalize(vmin=0,vmax=0.1)
ax_sp02.contourf(tm_sp02, freq_sp02, np.log10(np.abs(sp02)**2),
                 norm=norm, levels=256, cmap=cmap)

ax_sp02.text(0.99, 0.97, "spectrogram", color='white', ha='right', va='top',
              path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                            path_effects.Normal()],
              transform=ax_sp02.transAxes)

#mpl.colorbar.ColorbarBase(cb_sp02, cmap=cmap, norm=norm,
#boundaries=np.linspace(-0, 0.1, 17),
                          #orientation="vertical",
                          #label='$\log_{10}|Y/N|^2$')
#mpl.colorbar.ColorbarBase(cb_xsp, cmap=cmap, norm=norm,
                          #orientation="vertical",
                          #label='$\log_{10}|XY^*/N^2|$')
mpl.colorbar.ColorbarBase(cb_sp02, cmap=cmap, norm=norm,
                          orientation="vertical",
                          label='$\log_{10}|X/N|^2$',
                          ticks=np.linspace(np.log10(np.abs(sp01[freq_sp02 < freq_upper, :])**2).min(),
                                            np.log10(np.abs(sp01[freq_sp02 < freq_upper, :])**2).max(), 5))
# ---------------------------
# ---------------------------
# テスト信号 sig01 と sig02 のクロススペクトルのプロット
ax_xsp.set_xlim(tms, tme)
ax_xsp.set_xlabel('')
ax_xsp.tick_params(labelbottom=False)
# クロススペクトルの縦軸の範囲を2Hzからに変更
ax_xsp.set_ylim(2, freq_upper)
ax_xsp.set_ylabel('frequency\n(Hz)')

norm = mpl.colors.Normalize(vmin=np.log10(np.abs(xsp[freq_sp02 < freq_upper, :])).min(),
                            vmax=np.log10(np.abs(xsp[freq_sp02 < freq_upper, :])).max())

ax_xsp.contourf(tm_sp01, freq_sp01, np.log10(np.abs(xsp)),
                norm=norm, levels=256, cmap=cmap)

ax_xsp.text(0.99, 0.97, "cross-spectrum", color='white', ha='right', va='top',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                          path_effects.Normal()],
            transform=ax_xsp.transAxes)

mpl.colorbar.ColorbarBase(cb_xsp, cmap=cmap, norm=norm,
                          orientation="vertical",
                          label='$\log_{10}|XY^*/N^2|$')

# ---------------------------
# テスト信号 sig01 と sig02 のコヒーレンスのプロット
ax_coh.set_xlim(tms, tme)
ax_coh.set_xlabel('')
ax_coh.tick_params(labelbottom=False)
# コヒーレンスの縦軸の範囲を2Hzからに変更
ax_coh.set_ylim(2, freq_upper)
ax_coh.set_ylabel('frequency\n(Hz)')

norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
ax_coh.contourf(tm_sp01, freq_sp01, coh,
                norm=norm, levels=10, cmap=cmap)

ax_coh.text(0.99, 0.97, "coherence", color='white', ha='right', va='top',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                          path_effects.Normal()],
            transform=ax_coh.transAxes)

mpl.colorbar.ColorbarBase(cb_coh, cmap=cmap, norm=norm,
                          boundaries=np.linspace(0.8, 1, 5),
                          orientation="vertical",
                          label='coherence')

# ---------------------------
# テスト信号 sig01 と sig02 のフェイズのプロット
ax_phs.set_xlim(tms, tme)
ax_phs.set_xlabel('time (s)')
ax_phs.tick_params(labelbottom=True)
# フェイズの縦軸の範囲を2Hzからに変更
ax_phs.set_ylim(2, freq_upper)
ax_phs.set_ylabel('frequency\n(Hz)')

norm = mpl.colors.Normalize(vmin=-180.0, vmax=180.0)
cmap = mpl.cm.hsv

# 修正点：np.where()の代わりにnp.ma.masked_where()を使用してマスク
phs_masked = np.ma.masked_where(coh < 0.65, phs)
ax_phs.contourf(tm_sp01, freq_sp01, phs_masked,
                norm=norm, levels=30, cmap=cmap)

ax_phs.text(0.99, 0.97, "phase", color='white', ha='right', va='top',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                          path_effects.Normal()],
            transform=ax_phs.transAxes)

mpl.colorbar.ColorbarBase(cb_phs, cmap=cmap,
                          norm=norm,
                          boundaries=np.linspace(-180.0, 180.0, 17),
                          orientation="vertical",
                          label='phase (deg)')

# ファイルに保存
#out_fig_path = f"C:/Users/white/baipro/figure/001_phase/sweep/wavelet/figure_{channel1}_{channel2}.png"
#plt.savefig(out_fig_path, transparent=False)

st.pyplot(fig)
