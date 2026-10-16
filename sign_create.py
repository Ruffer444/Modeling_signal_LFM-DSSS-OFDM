import numpy as np
# from scipy.signal import chirp
import matplotlib.pyplot as plt
# Функция чтобы создать сигналы, пока без шума (lFM,HFM,OFDM,DSSS)
def make_signal(kind,fs=48e3,T=None,f0=4e3,f1=16e3,N_fft=512,
                N_cp=64, N_sym=8, mod_order=4,
                bits=(1, 0, 1, 1, 0, 0, 1, 0),
                sf=63,seed=0, real=False, f_c=12e3):

# kind - название метода создания ('lfm' | 'hfm' | 'ofdm' | 'dsss')
# fs - частота дискретизации, Гц
# T - длительность, с (для ЛЧМ)
# f0 - нижний порог частот, Гц (для ЛЧМ)
# f1 - верхний порог частоты, Гц (для ЛЧМ)

# N_fft - число поднесущих OFDM

# bits - информационные биты для расширения DSSS

# seed - зерно ГПСЧ (для воспроизводимости PN и QAM)
# real - (True) вещественная полоса пропускания на несущей f_c
#        (False) комплексная полоса пропускания iQ
# f_c - несущая частота для переноса Гц


    rng = np.random.default_rng(seed)

    match kind:
        case 'lfm':
            # t = np.arange(int(fs*T)) / fs - T / 2
            # bb = chirp(t, f0=f0,f1=f1,t1=T,method = 'linear').astype(complex)
            # print(f' Имя = {kind} , Длинна = {len(bb)} , Тип данных: {bb.dtype} , Максимум:{np.max(np.abs(bb))}')
            t = np.arange(int(fs*T)) / fs
            K = (f1 - f0) / T
            phase = 2*np.pi*(f0*t + 0.5*K*t**2)
            bb = np.exp(1j*phase)
            
        case 'hfm':
            t = np.arange(int(fs*T)) / fs
            # гиперболический чирп
            f0_, f1_ = f0, f1
            k = (f0_*f1_*T) / (f1_ - f0_)
            phase = -2*np.pi*k*np.log(1 - (f1_-f0_)/(f1_*T) * t)
            bb = np.exp(1j*phase)
            # t = np.arange(int(fs*T)) / fs - T / 2
            # bb = chirp(t, f0=f0,f1=f1,t1=T,method = 'hyperbolic').astype(complex)
            # print(f' Имя = {kind} , Длинна = {len(bb)} , Тип данных: {bb.dtype} , Максимум:{np.max(np.abs(bb))}')
        case 'dsss':
            pn = rng.choice([-1,1],size=sf)
            symbols = 1 - 2*np.array(bits, dtype = float) # 0 ->+1, 1->-1
            bb = np.kron(symbols, pn).astype(complex)
            # print(f' Имя = {kind} , Длинна = {len(bb)} , Тип данных: {bb.dtype} , Максимум:{np.max(np.abs(bb))}')

        case 'ofdm':
            match mod_order:
                case 2:
                    qam = (1 - 2 * rng.integers(0,2, size = (N_sym, N_fft))).astype(complex)

                case 4:
                    ang = np.pi/4 + np.pi/2 * rng.integers(0,4, size = (N_sym, N_fft))

                    qam = np.exp(1j * ang)

                case _:
                    raise ValueError("mod_order: 2 или 4")

            syms = np.fft.ifft(qam, axis=1)

            if N_cp > 0:
                syms_cp = np.concatenate(
                    [syms[:, -N_cp:], syms],
                axis=1
        )
            else:
                syms_cp = syms
            bb = syms_cp.ravel().astype(complex)
            bb /= np.sqrt(np.mean(np.abs(bb)**2))
            # print(f' Имя = {kind} , Длинна = {len(bb)} , Тип данных: {bb.dtype} , Максимум:{np.max(np.abs(bb))}')

        case _:
            raise ValueError(f"Неизвестный метод (kind): {kind}")
    if real:
        t_rf = np.arange(len(bb)) / fs
        return np.real(bb * np.exp(1j * 2*np.pi * f_c * t_rf))
    return bb


def sign_status_table(signals):
    """
    signals : список кортежей (name, x)
    """
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


def sign_plt(signals, fs=48e3, name='signal'):
    """Отрисовка сигнала: время, мгновенная частота, спектр."""
    x = np.asarray(signals)
    t = np.arange(len(x)) / fs

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
    plt.plot(t[1:], np.diff(np.unwrap(np.angle(x))) * fs / (2 * np.pi) / 1e3)
    plt.xlabel('t, с'); plt.ylabel('f, кГц'); plt.title('мгновенная частота'); plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    X = np.fft.fftshift(np.fft.fft(x))
    plt.plot(np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs)) / 1e3,
             20 * np.log10(np.abs(X) / np.abs(X).max() + 1e-12))
    plt.xlabel('f, кГц'); plt.ylabel('|X|, дБ'); plt.title('спектр')
    plt.ylim(-80, 5); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()