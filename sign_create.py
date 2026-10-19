import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


# Функция чтобы создать сигналы, пока без шума (LFM, HFM, OFDM, DSSS)
def make_signal(kind, fs=48e3, T=None, f0=4e3, f1=16e3, N_fft=512,
                N_cp=64, N_sym=8, mod_order=4,
                bits=(1, 0, 1, 1, 0, 0, 1, 0),
                sf=63, seed=0, real=False, f_c=12e3,
                data_bits=None, n_bits_chirp=100):
# kind - название метода создания ('lfm' | 'hfm' | 'ofdm' | 'dsss')
# fs - частота дискретизации, Гц
# T - длительность, с (для ЛЧМ)
# f0, f1 - нижний/верхний порог частот, Гц (для ЛЧМ)
# N_fft - число поднесущих OFDM
# bits - информационные биты для расширения DSSS (если data_bits=None)
# seed - зерно ГПСЧ (для воспроизводимости PN и QAM)
# real - (True) вещественная полоса на несущей f_c, (False) комплексный iQ
# f_c - несущая частота для переноса Гц
# data_bits - биты для внедрения в сигнал (0/1, np.uint8), переопределяют
#             bits для DSSS и заполняют QAM для OFDM, а для LFM/HFM —
#             BPSK-манипуляция n_bits_chirp сегментов
# n_bits_chirp - сколько бит кодируется в одном чирпе LFM/HFM

    rng = np.random.default_rng(seed)

    match kind:
        case 'lfm':
            t = np.arange(int(fs*T)) / fs
            K = (f1 - f0) / T
            phase = 2*np.pi*(f0*t + 0.5*K*t**2)
            bb = np.exp(1j*phase)
            if data_bits is not None:
                bb = _bpsk_on_chirp(bb, data_bits, n_bits_chirp)

        case 'hfm':
            t = np.arange(int(fs*T)) / fs
            k = (f0*f1*T) / (f1 - f0)
            phase = -2*np.pi*k*np.log(1 - (f1-f0)/(f1*T) * t)
            bb = np.exp(1j*phase)
            if data_bits is not None:
                bb = _bpsk_on_chirp(bb, data_bits, n_bits_chirp)

        case 'dsss':
            pn = rng.choice([-1, 1], size=sf)
            src = data_bits if data_bits is not None else bits
            symbols = 1 - 2*np.array(src, dtype=float)   # 0 ->+1, 1->-1
            bb = np.kron(symbols, pn).astype(complex)

        case 'ofdm':
            if data_bits is not None:
                qam = _bits_to_qam(data_bits, N_sym, N_fft, mod_order)
            else:
                match mod_order:
                    case 2:
                        qam = (1 - 2 * rng.integers(0, 2, size=(N_sym, N_fft))).astype(complex)
                    case 4:
                        ang = np.pi/4 + np.pi/2 * rng.integers(0, 4, size=(N_sym, N_fft))
                        qam = np.exp(1j * ang)
                    case _:
                        raise ValueError("mod_order: 2 или 4")

            syms = np.fft.ifft(qam, axis=1)
            if N_cp > 0:
                syms = np.concatenate([syms[:, -N_cp:], syms], axis=1)
            bb = syms.ravel().astype(complex)

        case _:
            raise ValueError(f"Неизвестный метод (kind): {kind}")

    # Единая нормировка RMS = 1
    bb = bb / np.sqrt(np.mean(np.abs(bb)**2))

    if real:
        return np.sqrt(2) * np.real(bb * np.exp(1j * 2*np.pi*f_c*np.arange(len(bb))/fs))
    return bb


# --- Вспомогательные функции для data_bits ---
def _bits_to_qam(data_bits, N_sym, N_fft, mod_order):
    # Раскладываем биты в QAM-созвездие, недостающие дополняем нулями
    b = np.asarray(data_bits, dtype=np.uint8).ravel()
    if mod_order == 2:
        bits_per_sym = 1
    elif mod_order == 4:
        bits_per_sym = 2
    else:
        raise ValueError("mod_order: 2 или 4")

    n_slots = N_sym * N_fft
    n_need = n_slots * bits_per_sym
    if len(b) > n_need:
        raise ValueError(f"data_bits: {len(b)} бит > ёмкости OFDM {n_need} бит")
    if len(b) < n_need:
        b = np.concatenate([b, np.zeros(n_need - len(b), dtype=np.uint8)])

    if mod_order == 2:
        # BPSK: 0 -> +1, 1 -> -1
        return (1 - 2*b).reshape(N_sym, N_fft).astype(complex)

    # QPSK: пара бит (b0,b1) -> k = 2*b0 + b1 -> exp(j*(pi/4 + pi/2*k))
    pairs = b.reshape(-1, 2)
    k = 2*pairs[:, 0].astype(int) + pairs[:, 1].astype(int)
    ang = np.pi/4 + np.pi/2 * k
    return np.exp(1j*ang).reshape(N_sym, N_fft)


def _bpsk_on_chirp(bb, data_bits, n_bits_chirp):
    # Разбиваем чирп на n_bits_chirp сегментов, в каждом ставим знак ±1
    b = np.asarray(data_bits, dtype=np.uint8).ravel()
    if len(b) > n_bits_chirp:
        raise ValueError(
            f"data_bits: {len(b)} бит > ёмкости чирпа {n_bits_chirp} бит. "
            f"Увеличьте T или n_bits_chirp."
        )
    if len(b) < n_bits_chirp:
        b = np.concatenate([b, np.zeros(n_bits_chirp - len(b), dtype=np.uint8)])

    out = bb.copy()
    n = len(bb)
    for i, bit in enumerate(b):
        start = i * n // n_bits_chirp
        end = (i + 1) * n // n_bits_chirp
        if bit:
            out[start:end] *= -1
    return out


def sign_status_table(signals):
    header = f'{"Имя":<20}{"Длина":>8}{"Тип":>14}{"Пик":>10}{"RMS":>10}{"PAPR,дБ":>10}'
    print(header)
    print('-' * len(header))
    for name, x in signals:
        x = np.asarray(x)
        mag = np.abs(x)
        peak = np.max(mag)
        rms = np.sqrt(np.mean(mag**2))
        papr = 10*np.log10(peak**2 / rms**2) if rms > 0 else float('inf')
        print(f'{name:<20}{len(x):>8}{str(x.dtype):>14}'
              f'{peak:>10.3f}{rms:>10.3f}{papr:>10.2f}')


def sign_plt(signals, fs=48e3, name='signal', block=True):
    x = np.asarray(signals)
    t = np.arange(len(x)) / fs
    analytic = x if np.iscomplexobj(x) else hilbert(x)

    plt.figure(figsize=(10, 9))

    plt.subplot(3, 1, 1)
    if np.iscomplexobj(x):
        plt.plot(t, x.real, label='Re')
        plt.plot(t, x.imag, label='Im')
        plt.plot(t, np.abs(x), 'k', alpha=0.6, label='|x|')
        plt.legend(fontsize=8)
    else:
        plt.plot(t, x)
    plt.title(name); plt.xlabel('t, с'); plt.ylabel('амплитуда'); plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(t[1:], np.diff(np.unwrap(np.angle(analytic))) * fs / (2 * np.pi) / 1e3)
    plt.xlabel('t, с'); plt.ylabel('f, кГц'); plt.title('мгновенная частота'); plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    X = np.fft.fftshift(np.fft.fft(x))
    plt.plot(np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs)) / 1e3,
             20 * np.log10(np.abs(X) / np.abs(X).max() + 1e-12))
    plt.xlabel('f, кГц'); plt.ylabel('|X|, дБ'); plt.title('спектр')
    plt.ylim(-80, 5); plt.grid(alpha=0.3)

    plt.tight_layout()
   