# analysis.py
# Расширенный анализ: распределение битов, корреляция, BER/PER,
# сила сигнала, сводная таблица с принятыми сообщениями.

import numpy as np
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from sign_create import make_signal
from create_insert_msg_to_signal import (
    bytes_to_bits, bits_to_bytes,
    demod_ofdm, demod_dsss, demod_chirp, receive,
    HEADER_LEN, NONCE_LEN, TAG_LEN, KEY,
)


# ============================================================
# 1. СИЛА СИГНАЛА И SNR
# ============================================================

def set_signal_power(x, rms_target=1.0):
    """Масштабирует сигнал так, чтобы его RMS = rms_target."""
    x = np.asarray(x)
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms == 0:
        return x.copy()
    return x * (rms_target / rms)


def measure_snr(x_clean, x_noisy):
    """Фактическое SNR по известному чистому сигналу."""
    p_sig = np.mean(np.abs(x_clean) ** 2)
    p_noise = np.mean(np.abs(x_noisy - x_clean) ** 2)
    return 10 * np.log10(p_sig / p_noise) if p_noise > 0 else float('inf')


def add_noise_with_check(x_clean, snr_db, seed=0):
    """AWGN: шум масштабируется под сигнал. Возвращает (y, snr_measured)."""
    rng = np.random.default_rng(seed)
    p_sig = np.mean(np.abs(x_clean) ** 2)
    p_noise = p_sig / 10 ** (snr_db / 10)
    noise = np.sqrt(p_noise / 2) * (
        rng.standard_normal(len(x_clean)) + 1j * rng.standard_normal(len(x_clean))
    )
    y = x_clean + noise
    return y, measure_snr(x_clean, y)


def add_awgn_fixed_noise(x, noise_power, seed=0):
    """AWGN с ФИКСИРОВАННОЙ мощностью шума (не зависит от сигнала).
    SNR = p_sig / noise_power — меняется через силу сигнала."""
    rng = np.random.default_rng(seed)
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    )
    y = x + noise
    p_sig = np.mean(np.abs(x) ** 2)
    snr = 10 * np.log10(p_sig / noise_power) if noise_power > 0 else float('inf')
    return y, snr


# ============================================================
# 2. КАРТА РАСПРЕДЕЛЕНИЯ БИТОВ В СИГНАЛЕ
# ============================================================

def plot_bit_distribution(kind, fs, bits, **kwargs):
    """Показывает, как биты «сидят» в сигнале для выбранного метода.
    Padding (нули-заполнители) отмечается отдельно."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if kind == 'ofdm':
        _plot_bit_ofdm(bits, **kwargs)
    elif kind == 'dsss':
        _plot_bit_dsss(bits, **kwargs)
    elif kind in ('lfm', 'hfm'):
        _plot_bit_chirp(kind, fs, bits, **kwargs)
    else:
        raise ValueError(f'kind={kind}')


def _plot_bit_ofdm(bits, N_fft=512, N_sym=8, N_cp=64, mod_order=4, **kwargs):
    bps = 1 if mod_order == 2 else 2
    n_need = N_sym * N_fft * bps
    grid_idx = np.full((N_sym, N_fft), -1, dtype=int)
    grid_val = np.full((N_sym, N_fft), np.nan)

    for s in range(N_sym):
        for k in range(N_fft):
            base = (s * N_fft + k) * bps
            if base + bps - 1 < len(bits):
                grid_idx[s, k] = base
                if bps == 1:
                    grid_val[s, k] = 1 - 2 * bits[base]
                else:
                    b0, b1 = bits[base], bits[base + 1]
                    kq = 2 * b0 + b1
                    grid_val[s, k] = np.pi / 4 + np.pi / 2 * kq

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    im0 = ax[0].imshow(grid_idx, aspect='auto', cmap='viridis',
                       origin='lower', interpolation='nearest')
    ax[0].set_title(f'Индекс первого бита (bps={bps})\n'
                    f'Реальных: {len(bits)} из {n_need} (остальное — padding)')
    ax[0].set_xlabel('Поднесущая k'); ax[0].set_ylabel('Символ s')
    plt.colorbar(im0, ax=ax[0], label='bit_idx')

    if mod_order == 2:
        im1 = ax[1].imshow(grid_val, aspect='auto', cmap='RdBu',
                           vmin=-1, vmax=1, origin='lower')
        ax[1].set_title('BPSK: значение бита (±1)')
    else:
        im1 = ax[1].imshow(grid_val, aspect='auto', cmap='hsv',
                           vmin=-np.pi, vmax=np.pi, origin='lower')
        ax[1].set_title('QPSK: фаза (радианы)')
    ax[1].set_xlabel('Поднесущая k'); ax[1].set_ylabel('Символ s')
    plt.colorbar(im1, ax=ax[1])

    fig.suptitle('OFDM: распределение битов (символ × поднесущая)')
    plt.tight_layout()


def _plot_bit_dsss(bits, sf=63, seed=0, **kwargs):
    signs = 1 - 2 * bits.astype(float)
    rng = np.random.default_rng(seed)
    pn = rng.choice([-1, 1], size=sf)
    chips = np.kron(signs, pn)
    n = len(chips)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    t = np.arange(n)

    ax[0].plot(t, chips, '.-', ms=2, lw=0.4)
    for i in range(len(bits) + 1):
        ax[0].axvline(i * sf - 0.5, color='r', ls='--', alpha=0.25, lw=0.5)
    ax[0].set_title(f'DSSS: {len(bits)} бит × {sf} чипов = {n} чипов')
    ax[0].set_xlabel('Индекс чипа'); ax[0].set_ylabel('Значение ±1')
    ax[0].set_ylim(-1.4, 1.4); ax[0].grid(alpha=0.3)

    bit_val = np.repeat(signs, sf)
    bit_idx = np.repeat(np.arange(len(bits)), sf)
    sc = ax[1].scatter(t, bit_idx, c=bit_val, cmap='RdBu',
                       vmin=-1, vmax=1, s=4)
    ax[1].set_title('Какой бит в каком чипе (цвет = значение бита)')
    ax[1].set_xlabel('Индекс чипа'); ax[1].set_ylabel('Индекс бита')
    plt.colorbar(sc, ax=ax[1], label='бит ±1')

    plt.tight_layout()

    # Проверка ортогональности PN
    ac = np.correlate(pn, pn, mode='full') / sf
    side = np.abs(np.delete(ac, sf - 1))
    print(f'  PN(sf={sf}): пик ACF = {ac[sf-1]:.3f}, '
          f'макс. бок. лепесток = {side.max():.3f}')


def _plot_bit_chirp(kind, fs, bits, T=100e-3, n_bits_chirp=1000,
                    f0=4e3, f1=16e3, **kwargs):
    n = int(fs * T)
    tx = make_signal(kind, fs=fs, T=T, f0=f0, f1=f1)

    seg_val = np.full(n, np.nan)
    for i in range(n_bits_chirp):
        start = i * n // n_bits_chirp
        end = (i + 1) * n // n_bits_chirp
        if i < len(bits):
            seg_val[start:end] = 1 - 2 * bits[i]
        else:
            seg_val[start:end] = 0.0  # padding

    t_ms = np.arange(n) / fs * 1e3

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    ax[0].plot(t_ms, np.real(tx), lw=0.5)
    for i in range(n_bits_chirp + 1):
        ax[0].axvline(i * T * 1e3 / n_bits_chirp,
                      color='r', ls='--', alpha=0.15, lw=0.4)
    ax[0].set_title(f'{kind.upper()}: {n_bits_chirp} сегментов на чирпе')
    ax[0].set_xlabel('t, мс'); ax[0].set_ylabel('Re')
    ax[0].grid(alpha=0.3)

    ax[1].plot(t_ms, seg_val, lw=0.6)
    ax[1].set_title('Значение бита в каждом сегменте (±1, 0 = padding)')
    ax[1].set_xlabel('t, мс'); ax[1].set_ylabel('бит')
    ax[1].set_yticks([-1, 0, 1]); ax[1].grid(alpha=0.3)

    N_show = min(20, n_bits_chirp)
    mask = np.arange(n) < int(N_show * n / n_bits_chirp)
    ax[2].plot(t_ms[mask], seg_val[mask], 'o-', lw=1.5, ms=3)
    ax[2].set_title(f'Первые {N_show} сегментов (крупно)')
    ax[2].set_xlabel('t, мс'); ax[2].set_ylabel('бит')
    ax[2].set_yticks([-1, 0, 1]); ax[2].grid(alpha=0.3)

    plt.tight_layout()


# ============================================================
# 3. КОРРЕЛЯЦИЯ TX/RX БИТОВ
# ============================================================

def plot_bit_correlation(bits_tx, bits_rx, title=''):
    """TX vs RX: карта ошибок + автокорреляция ошибок.
    Периодические пики в автокорреляции → ICI/ISI."""
    n = min(len(bits_tx), len(bits_rx))
    err = bits_tx[:n] != bits_rx[:n]
    ber = float(np.mean(err))

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    N_show = min(200, n)
    t = np.arange(N_show)
    ax[0].step(t, bits_tx[:N_show], where='mid', label='TX', lw=1)
    ax[0].step(t, bits_rx[:N_show] + 1.2, where='mid', label='RX (+1.2)', lw=1)
    ax[0].set_title(f'Первые {N_show} бит  (BER = {ber:.4f})')
    ax[0].set_xlabel('Индекс бита'); ax[0].set_ylabel('бит')
    ax[0].set_yticks([0, 1, 1.2, 2.2]); ax[0].set_yticklabels(['0', '1', '0', '1'])
    ax[0].legend(fontsize=8, loc='upper right'); ax[0].grid(alpha=0.3)

    n_cols = 100
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    eg = np.zeros(n_rows * n_cols, dtype=int)
    eg[:n] = err.astype(int)
    eg = eg.reshape(n_rows, n_cols)
    ax[1].imshow(eg, aspect='auto', cmap='Reds', vmin=0, vmax=1,
                 interpolation='nearest')
    ax[1].set_title(f'Карта ошибок: {int(err.sum())} / {n}')
    ax[1].set_xlabel('Позиция внутри строки (100 бит/строка)')
    ax[1].set_ylabel('Строка')

    if err.std() > 0:
        err_signed = err.astype(float) * 2 - 1
        max_lag = min(100, n // 2)
        ac = np.correlate(err_signed, err_signed, mode='full') / n
        lags = np.arange(-n + 1, n)
        mask = np.abs(lags) <= max_lag
        ax[2].stem(lags[mask], ac[mask], basefmt=' ')
        ax[2].set_title('Автокорреляция ошибок '
                        '(пики при τ≠0 → периодические ошибки / ICI)')
        ax[2].set_xlabel('Лаг τ (бит)'); ax[2].set_ylabel('r(τ)')
        ax[2].grid(alpha=0.3)
    else:
        ax[2].text(0.5, 0.5, 'Нет ошибок', ha='center', va='center',
                   transform=ax[2].transAxes, fontsize=14)

    fig.suptitle(title or 'TX/RX: где и как ошибаемся')
    plt.tight_layout()


# ============================================================
# 4. BER / PER — СВИТ ПО SNR
# ============================================================

def _make_fixed_packet(text, key=KEY):
    """Пакет с детерминированным nonce — для воспроизводимости BER."""
    nonce = bytes(NONCE_LEN)
    ct = ChaCha20Poly1305(key).encrypt(nonce, text.encode('utf-8'), None)
    body = nonce + ct
    return len(body).to_bytes(HEADER_LEN, 'big') + body


def _demod(kind, rx, fs, **kw):
    if kind == 'ofdm':
        return demod_ofdm(rx, fs, kw['N_fft'], kw['N_cp'],
                          kw['N_sym'], kw['mod_order'])
    if kind == 'dsss':
        return demod_dsss(rx, kw.get('sf', 63), seed=kw.get('seed', 0))
    if kind in ('lfm', 'hfm'):
        return demod_chirp(rx, fs, kw['T'], kw['f0'], kw['f1'],
                           kind, kw['n_bits_chirp'])
    raise ValueError(kind)


def run_ber_trial(kind, bits_tx, text, fs, snr_db, seed=0, **kw):
    """Один прогон: tx → канал → демод. Возвращает BER/PER/bits_rx."""
    tx = make_signal(kind, fs=fs, data_bits=bits_tx, **kw)
    rx, snr_meas = add_noise_with_check(tx, snr_db, seed=seed)
    bits_rx = _demod(kind, rx, fs, **kw)

    n = min(len(bits_tx), len(bits_rx))
    ber = float(np.mean(bits_tx[:n] != bits_rx[:n])) if n > 0 else 1.0
    text_rx = receive(kind, rx, fs, len(bits_tx), **kw)
    ok = (text_rx == text)

    return {'ber': ber, 'ok': ok, 'text_rx': text_rx, 'snr_meas': snr_meas,
            'bits_rx': bits_rx, 'tx': tx, 'rx': rx}


# ============================================================
# 5. RAW-ДЕКОДИРОВАНИЕ (для диагностики — без проверки тега)
# ============================================================

def decrypt_raw(nonce, ct_with_tag, key=KEY):
    """ChaCha20 без проверки Poly1305.
    Возвращает сырые байты plaintext (мусор, если биты побиты).
    Используется ТОЛЬКО для диагностики — в реальном приёмнике
    тег Poly1305 обязателен."""
    ct = ct_with_tag[:-TAG_LEN]
    # ChaCha20Poly1305: counter=0 → ключ Poly1305, counter=1 → шифртекст
    nonce16 = b'\x01\x00\x00\x00' + nonce
    cipher = Cipher(algorithms.ChaCha20(key, nonce16), mode=None)
    return cipher.encryptor().update(ct)


def receive_raw(kind, rx, fs, n_bits_hint, key=KEY, **kwargs):
    """Как receive(), но дополнительно возвращает raw-декодированный
    текст (без тега). Возвращает (text_ok, text_raw, tag_ok, bits_rx)."""
    bits = _demod(kind, rx, fs, **kwargs)
    if len(bits) < HEADER_LEN * 8:
        return None, None, False, bits
    body_len = int.from_bytes(bits_to_bytes(bits[:HEADER_LEN * 8]), 'big')
    total_bits = (HEADER_LEN + body_len) * 8
    if len(bits) < total_bits:
        return None, None, False, bits

    data = bits_to_bytes(bits[:total_bits])
    body = data[HEADER_LEN:HEADER_LEN + body_len]
    nonce = body[:NONCE_LEN]
    ct = body[NONCE_LEN:]

    try:
        text_ok = ChaCha20Poly1305(key).decrypt(nonce, ct, None).decode('utf-8')
        tag_ok = True
    except Exception:
        text_ok = None
        tag_ok = False

    try:
        text_raw = decrypt_raw(nonce, ct, key).decode('utf-8')
    except Exception:
        text_raw = None

    return text_ok, text_raw, tag_ok, bits


# ============================================================
# 6. ТАБЛИЦА СРАВНЕНИЯ МЕТОДОВ (с сообщениями)
# ============================================================

def results_table(methods, text, fs, snr_list=None, n_trials=5,
                  base_seed=0, key=KEY, verbose=True):
    """Сравнительная таблица. methods = {'ofdm': {...}, 'dsss': {...}, ...}."""
    if snr_list is None:
        snr_list = list(np.arange(-10, 22, 2))

    packet = _make_fixed_packet(text, key=key)
    bits_tx = bytes_to_bits(packet)

    txs = {name: make_signal(name, fs=fs, data_bits=bits_tx, **kw)
           for name, kw in methods.items()}

    results = {name: [] for name in methods}
    for snr in snr_list:
        for name, kw in methods.items():
            bers, oks = [], 0
            texts_ok, texts_raw = [], []
            for t in range(n_trials):
                rx, _ = add_noise_with_check(txs[name], snr, seed=base_seed + t)
                bits_rx = _demod(name, rx, fs, **kw)
                n = min(len(bits_tx), len(bits_rx))
                bers.append(float(np.mean(bits_tx[:n] != bits_rx[:n])))

                t_ok, t_raw, _tag_ok, _ = receive_raw(
                    name, rx, fs, len(bits_tx), key=key, **kw)
                oks += (t_ok == text)
                texts_ok.append(t_ok)
                texts_raw.append(t_raw)

            results[name].append({
                'snr': snr,
                'ber': float(np.mean(bers)),
                'per': 1.0 - oks / n_trials,
                'texts_ok': texts_ok,
                'texts_raw': texts_raw,
            })

    if verbose:
        _print_table(methods, results, snr_list, text, len(bits_tx), n_trials)
        _print_messages(methods, results, snr_list, text)
    return results


def _print_table(methods, results, snr_list, text, n_bits, n_trials):
    names = list(methods.keys())
    w = 18
    total = 8 + len(names) * w
    print()
    print('=' * total)
    print(f' СРАВНЕНИЕ МЕТОДОВ ПРИ РАЗНЫХ SNR  (trials/SNR = {n_trials})')
    print(f' Сообщение: {text!r}   |   Пакет: {n_bits} бит')
    print('=' * total)

    h1 = f'{"SNR,дБ":>7} |'
    for name in names:
        h1 += f'{name.upper():^{w}}|'
    print(h1)
    h2 = ' ' * 8 + '|'
    for _ in names:
        h2 += f'{"BER    PER   Статус":^{w}}|'
    print(h2)
    print('-' * total)

    for i, snr in enumerate(snr_list):
        row = f'{snr:>7.1f} |'
        for name in names:
            r = results[name][i]
            ber, per = r['ber'], r['per']
            status = 'OK' if per == 0.0 else ('FAIL' if per >= 0.999 else 'PART')
            row += f' {ber:>5.3f}  {per:>4.2f}  {status:<4} |'
        print(row)
    print('=' * total)


def _fmt_cell(text, max_len=24):
    """Компактная ячейка: 'Текст', '<None>', либо 'Text…'."""
    if text is None:
        return '<None>'
    s = repr(text)
    if len(s) > max_len:
        s = s[:max_len - 2] + '…'
    return s


def _pick_rep(r, expected):
    """Выбрать представителя: если всё ОК — вернуть expected;
    если были фейлы — вернуть первый фейл, чтобы была видна деградация."""
    if all(t == expected for t in r['texts_ok']):
        return expected, '✓'
    for tok, traw in zip(r['texts_ok'], r['texts_raw']):
        if tok != expected:
            if traw is None:
                return None, '?'
            if traw == expected:
                return traw, '!'   # тег упал, но текст цел
            return traw, '✗'       # текст битый
    return None, '?'


def _print_messages(methods, results, snr_list, expected_text, max_len=22):
    names = list(methods.keys())
    w = max_len + 4
    total = 8 + len(names) * (w + 2)

    print()
    print('=' * total)
    print(' ПРИНЯТЫЕ СООБЩЕНИЯ (raw-декодирование ChaCha20, БЕЗ тега Poly1305)')
    print(f' Ожидалось: {expected_text!r}')
    print('  Легенда:  ✓ тег OK + текст совпал   '
          '!  тег упал, текст цел   '
          '✗  текст битый   ?  не декодируется')
    print('=' * total)

    h = f'{"SNR,дБ":>7} |'
    for name in names:
        h += f'{name.upper():^{w}}|'
    print(h)
    print('-' * total)

    for i, snr in enumerate(snr_list):
        row = f'{snr:>7.1f} |'
        for name in names:
            disp, mark = _pick_rep(results[name][i], expected_text)
            cell = _fmt_cell(disp, max_len)
            row += f' {mark} {cell:<{w-2}} |'
        print(row)
    print('=' * total)


# ============================================================
# 7. ГРАФИК BER/PER vs SNR (несколько методов)
# ============================================================

def plot_ber_comparison(methods, text, fs, snr_list=None, n_trials=5,
                        base_seed=0):
    results = results_table(methods, text, fs, snr_list=snr_list,
                            n_trials=n_trials, base_seed=base_seed,
                            verbose=False)
    snrs = [r['snr'] for r in next(iter(results.values()))]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in methods:
        bers = [max(r['ber'], 1e-6) for r in results[name]]
        pers = [r['per'] for r in results[name]]
        ax[0].semilogy(snrs, bers, 'o-', label=name)
        ax[1].plot(snrs, pers, 'o-', label=name)

    ax[0].set_xlabel('SNR, дБ'); ax[0].set_ylabel('BER')
    ax[0].set_title('BER vs SNR'); ax[0].grid(True, which='both', alpha=0.3)
    ax[0].legend()

    ax[1].set_xlabel('SNR, дБ'); ax[1].set_ylabel('PER')
    ax[1].set_title('PER vs SNR'); ax[1].set_ylim(-0.05, 1.05)
    ax[1].grid(alpha=0.3); ax[1].legend()
    plt.tight_layout()


# ============================================================
# 8. ДЕМО: ВЛИЯНИЕ СИЛЫ СИГНАЛА ПРИ ФИКС. ШУМЕ
# ============================================================

def demo_signal_power(kind, text, fs, snr_ref_db=20, rms_list=None,
                      seed=0, **kw):
    """Фикс. шум, меняем амплитуду → меняется SNR и BER."""
    if rms_list is None:
        rms_list = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    packet = _make_fixed_packet(text)
    bits_tx = bytes_to_bits(packet)
    tx_ref = make_signal(kind, fs=fs, data_bits=bits_tx, **kw)
    p_noise = np.mean(np.abs(tx_ref) ** 2) / 10 ** (snr_ref_db / 10)

    print()
    print(f'--- Влияние силы сигнала: {kind.upper()} ---')
    print(f'    Шум фиксирован: p_noise = {p_noise:.3e} '
          f'(SNR={snr_ref_db} дБ при RMS=1)')
    print(f'{"RMS":>8}{"SNR задано":>12}{"SNR факт,дБ":>14}'
          f'{"BER":>10}{"Статус":>8}')
    print('-' * 54)

    for rms in rms_list:
        tx = set_signal_power(tx_ref, rms)
        rx, snr_meas = add_awgn_fixed_noise(tx, p_noise, seed=seed)
        bits_rx = _demod(kind, rx, fs, **kw)
        n = min(len(bits_tx), len(bits_rx))
        ber = float(np.mean(bits_tx[:n] != bits_rx[:n]))
        text_rx = receive(kind, rx, fs, len(bits_tx), **kw)
        status = 'OK' if text_rx == text else 'FAIL'
        snr_target = snr_ref_db + 20 * np.log10(rms)
        print(f'{rms:>8.3f}{snr_target:>12.2f}{snr_meas:>14.2f}'
              f'{ber:>10.4f}{status:>8}')