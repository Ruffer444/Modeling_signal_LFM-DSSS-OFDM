import numpy as np
import matplotlib.pyplot as plt

# Доступные окна
WINDOW_NAMES = ('rect', 'hamming', 'hann', 'blackman', 'kaiser')


def get_window(name, N):
    # Возвращает окно длины N по имени.
    # rect     — прямоугольное (без окна), узкий главный лепесток,
    #            высокие боковые (-13 дБ)
    # hamming  — классика для радара/спектра, боковые -43 дБ
    # hann     — мягче hamming, боковые -31 дБ, чуть шире главный
    # blackman — очень низкие боковые (-58 дБ), самый широкий главный
    # kaiser   — настраиваемый (beta=8), компромисс
    if name == 'rect':     return np.ones(N)
    if name == 'hamming':  return np.hamming(N)
    if name == 'hann':     return np.hanning(N)
    if name == 'blackman': return np.blackman(N)
    if name == 'kaiser':   return np.kaiser(N, 8.0)
    raise ValueError(f'Неизвестное окно: {name}')


def plot_windows(names=WINDOW_NAMES, N=256):
    # Временные реализации + АЧХ окон — видно, чем они отличаются
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    t = np.arange(N)
    for name in names:
        w = get_window(name, N)
        ax[0].plot(t, w, label=name)

        W = np.abs(np.fft.fftshift(np.fft.fft(w)))
        W_db = 20 * np.log10(W / W.max() + 1e-12)
        f = np.fft.fftshift(np.fft.fftfreq(N))
        ax[1].plot(f, W_db, label=name)

    ax[0].set_title('Окна во времени')
    ax[0].set_xlabel('n'); ax[0].set_ylabel('w[n]')
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].set_title('АЧХ окон (нормированная), дБ')
    ax[1].set_xlabel('Норм. частота'); ax[1].set_ylabel('дБ')
    ax[1].set_ylim(-120, 5)
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show(block=False)


def plot_spectrum_windows(x, fs, names=WINDOW_NAMES,
                          title='Спектр с разными окнами'):
    # Один и тот же сигнал, но перед FFT умножен на разные окна.
    # Чем ниже боковые лепестки — тем чище спектр (но главный лепесток шире).
    fig, ax = plt.subplots(figsize=(11, 4))
    for name in names:
        w = get_window(name, len(x))
        X = np.fft.fftshift(np.fft.fft(x * w))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))
        X_db = 20 * np.log10(np.abs(X) / np.abs(X).max() + 1e-12)
        ax.plot(f / 1e3, X_db, label=name, alpha=0.8)

    ax.set_xlabel('f, кГц'); ax.set_ylabel('|X|, дБ')
    ax.set_title(title)
    ax.set_ylim(-100, 5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    # plt.show(block=False)


def compare_windows_on_chirp(fs, T=100e-3, f0=4e3, f1=16e3,
                             names=WINDOW_NAMES):
    # Сжатие чирпа с разными окнами. Ключевая метрика:
    #   - 1-й боковой лепесток (чем ниже, тем лучше видно слабые цели)
    #   - ширина главного лепестка -3 дБ (чем уже, тем лучше разрешение)
    # Импорт make_signal здесь, чтобы не тащить sign_create в модуль окон
    from sign_create import make_signal

    tx = make_signal('lfm', fs=fs, f0=f0, f1=f1, T=T)
    N = len(tx)

    print()
    print('--- Сравнение окон по сжатому чирпу (без шума) ---')
    print(f'{"Окно":<12}{"1-й бок. леп., дБ":>22}{"Ширина -3дБ, отсч.":>22}')
    print('-' * 56)

    fig, ax = plt.subplots(figsize=(11, 5))

    for name in names:
        w = get_window(name, N)
        mf = np.correlate(tx, tx * w, mode='full')
        mf_abs = np.abs(mf) / np.abs(mf).max()
        peak = int(np.argmax(mf_abs))

        # Ширина -3 дБ
        thresh = 1.0 / np.sqrt(2)
        i = peak
        while i < len(mf_abs) - 1 and mf_abs[i + 1] > thresh:
            i += 1
        width_3db = 2 * (i - peak)

        # Первый боковой лепесток
        i = peak
        while i < len(mf_abs) - 1 and mf_abs[i + 1] <= mf_abs[i]:
            i += 1
        hi = min(len(mf_abs), peak + 500)
        sll_db = 20 * np.log10(mf_abs[i:hi].max() + 1e-12) if hi > i else float('nan')

        print(f'{name:<12}{sll_db:>22.1f}{width_3db:>22d}')

        lo_show = max(0, peak - 100)
        hi_show = min(len(mf_abs), peak + 100)
        ax.plot(np.arange(lo_show, hi_show) - peak,
                20 * np.log10(mf_abs[lo_show:hi_show] + 1e-12),
                label=name, alpha=0.8)

    ax.set_xlabel('Отсчёт от пика'); ax.set_ylabel('|MF|, дБ')
    ax.set_title('Сжатый чирп: главный и боковые лепестки')
    ax.set_ylim(-100, 5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    # plt.show(block=False)