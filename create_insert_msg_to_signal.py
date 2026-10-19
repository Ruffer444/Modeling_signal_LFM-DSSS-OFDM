import os
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag
import matplotlib.pyplot as plt
from sign_create import make_signal

# Ключ 32 байта. Для теста фиксированный, чтобы BER/PER воспроизводились.
# В реальной системе ключ НИКОГДА не хардкодится — только из KDF/хранилища.
KEY = bytes(range(32))

NONCE_LEN = 12
TAG_LEN = 16
HEADER_LEN = 2        # длина тела пакета в байтах (big-endian)
MIN_SEG_SAMPLES = 10  # минимум отсчётов на один BPSK-сегмент LFM/HFM


# --- Биты <-> байты ---

def bytes_to_bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits):
    b = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(b)) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(b).tobytes()


# --- Шифрование ChaCha20-Poly1305 ---

def encrypt_message(text, key=KEY):
    # Возвращает (nonce, ciphertext_with_tag)
    nonce = os.urandom(NONCE_LEN)
    ct = ChaCha20Poly1305(key).encrypt(nonce, text.encode('utf-8'), None)
    return nonce, ct


def decrypt_message(nonce, ct, key=KEY):
    # Возвращает str, либо None если пакет битый/подменён
    try:
        return ChaCha20Poly1305(key).decrypt(nonce, ct, None).decode('utf-8')
    except InvalidTag:
        return None


# --- Пакет [длина 2 байта][nonce 12][ct + tag] ---

def pack_packet(text, key=KEY):
    nonce, ct = encrypt_message(text, key)
    body = nonce + ct
    header = len(body).to_bytes(HEADER_LEN, 'big')
    return header + body


def unpack_packet(data, key=KEY):
    # data : bytes
    if len(data) < HEADER_LEN + NONCE_LEN + TAG_LEN:
        return None
    body_len = int.from_bytes(data[:HEADER_LEN], 'big')
    if body_len < NONCE_LEN + TAG_LEN:
        return None
    body = data[HEADER_LEN:HEADER_LEN + body_len]
    if len(body) != body_len:
        return None
    nonce = body[:NONCE_LEN]
    ct = body[NONCE_LEN:]
    return decrypt_message(nonce, ct, key)


# --- Ввод сообщения ---

def read_user_message():
    text = input('Введите сообщение (например "20,20,-20"): ').strip()
    if not text:
        text = '20,20,-20'
        print(f'Пустой ввод — по умолчанию: {text}')
    return text


# --- Канал ---

def apply_awgn(x, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    p_sig = np.mean(np.abs(x)**2)
    p_noise = p_sig / 10**(snr_db/10)
    noise = np.sqrt(p_noise/2) * (rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    return x + noise


# --- Демодуляция ---

def demod_ofdm(rx, fs, N_fft, N_cp, N_sym, mod_order):
    # rx -> биты (все доступные)
    sym_len = N_fft + N_cp
    if len(rx) < sym_len * N_sym:
        raise ValueError('demod_ofdm: сигнал короче одного кадра')
    frames = rx[:sym_len * N_sym].reshape(N_sym, sym_len)
    qam = np.fft.fft(frames[:, N_cp:], axis=1)

    if mod_order == 2:
        bits = (np.real(qam) < 0).astype(np.uint8)   # BPSK
    elif mod_order == 4:
        ang = np.angle(qam.ravel())
        k = (np.round((ang - np.pi/4) / (np.pi/2)).astype(int)) % 4
        b0 = (k // 2).astype(np.uint8)
        b1 = (k % 2).astype(np.uint8)
        bits = np.stack([b0, b1], axis=1).ravel()
    else:
        raise ValueError('mod_order: 2 или 4')
    return bits


def demod_dsss(rx, sf, seed=0):
    # rx -> биты (все доступные символы)
    n_sym = len(rx) // sf
    if n_sym == 0:
        return np.zeros(0, dtype=np.uint8)
    chips = rx[:n_sym * sf].reshape(n_sym, sf)
    rng = np.random.default_rng(seed)
    pn = rng.choice([-1, 1], size=sf)   # та же PN, что у передатчика
    soft = np.real(chips @ pn)           # корреляция с PN
    return (soft < 0).astype(np.uint8)   # <0 -> бит 1, >0 -> бит 0


def demod_chirp(rx, fs, T, f0, f1, kind, n_bits_chirp):
    # BPSK на сегментах чирпа: де-чирп -> знак суммы по сегменту
    n = int(fs * T)
    t = np.arange(n) / fs
    if kind == 'lfm':
        K = (f1 - f0) / T
        ref = np.exp(1j * 2*np.pi*(f0*t + 0.5*K*t**2))
    else:  # hfm
        k = (f0*f1*T) / (f1 - f0)
        ref = np.exp(-1j * 2*np.pi*k*np.log(1 - (f1-f0)/(f1*T) * t))

    dechirped = rx[:n] * np.conj(ref)
    bits = np.zeros(n_bits_chirp, dtype=np.uint8)
    for i in range(n_bits_chirp):
        start = i * n // n_bits_chirp
        end = (i + 1) * n // n_bits_chirp
        bits[i] = 1 if np.real(dechirped[start:end].sum()) < 0 else 0
    return bits


# --- Верхний уровень ---

def _capacity_bits(kind, fs, **kwargs):
    # Сколько бит максимум можно вытащить из одного сигнала данного вида
    if kind == 'ofdm':
        N_fft = kwargs.get('N_fft', 512)
        N_sym = kwargs.get('N_sym', 8)
        mod_order = kwargs.get('mod_order', 4)
        return N_sym * N_fft * (1 if mod_order == 2 else 2)
    if kind == 'dsss':
        # длина сигнала = len(data_bits) * sf, но заранее не знаем
        return None
    if kind in ('lfm', 'hfm'):
        return kwargs.get('n_bits_chirp', 100)
    raise ValueError(f'kind={kind}')


def transmit(kind, text, fs, key=KEY, **kwargs):
    # text -> зашифрованный пакет -> биты -> сигнал
    packet = pack_packet(text, key)
    bits = bytes_to_bits(packet)

    cap = _capacity_bits(kind, fs, **kwargs)
    if cap is not None and len(bits) > cap:
        raise ValueError(
            f'{kind}: пакет {len(bits)} бит > ёмкости {cap} бит. '
            f'Сократите сообщение или увеличьте параметры сигнала.'
        )

    tx = make_signal(kind, fs=fs, data_bits=bits, **kwargs)
    return tx, len(bits)


def receive(kind, rx, fs, n_bits_hint, key=KEY, **kwargs):
    # сигнал -> биты -> пакет -> строка (или None)
    if kind == 'ofdm':
        bits = demod_ofdm(rx, fs,
                          kwargs['N_fft'], kwargs['N_cp'],
                          kwargs['N_sym'], kwargs['mod_order'])
    elif kind == 'dsss':
        bits = demod_dsss(rx, kwargs.get('sf', 63), seed=kwargs.get('seed', 0))
    elif kind in ('lfm', 'hfm'):
        bits = demod_chirp(rx, fs,
                           kwargs['T'], kwargs['f0'], kwargs['f1'],
                           kind, kwargs['n_bits_chirp'])
    else:
        raise ValueError(f'kind={kind}')

    # Обрезаем по фактической длине пакета из заголовка
    if len(bits) < HEADER_LEN * 8:
        return None
    body_len = int.from_bytes(bits_to_bytes(bits[:HEADER_LEN*8]), 'big')
    total_bits = (HEADER_LEN + body_len) * 8
    if len(bits) < total_bits:
        return None

    data = bits_to_bytes(bits[:total_bits])
    return unpack_packet(data, key)


def run_link(kind, text, fs, snr_db=20, seed=0, key=KEY, **kwargs):
    # Полный цикл: передача -> канал -> приём -> расшифровка -> печать
    try:
        tx, n_bits = transmit(kind, text, fs, key=key, **kwargs)
    except ValueError as e:
        print(f'[{kind:>4}] передача невозможна: {e}')
        return {'kind': kind, 'ok': False, 'reason': str(e)}

    rx = apply_awgn(tx, snr_db, seed=seed)
    text_rx = receive(kind, rx, fs, n_bits, key=key, **kwargs)

    if text_rx is None:
        print(f'[{kind:>4}] пакет повреждён (SNR={snr_db} дБ)')
        result = {'kind': kind, 'ok': False, 'reason': 'packet corrupted'}
    elif text_rx != text:
        print(f'[{kind:>4}] принято с искажением: {text_rx!r}')
        result = {'kind': kind, 'ok': False, 'reason': 'mismatch', 'text': text_rx}
    else:
        print(f'[{kind:>4}] OK  (SNR={snr_db} дБ): {text_rx!r}')
        result = {'kind': kind, 'ok': True, 'text': text_rx}

    return result

def run_link_packet(kind, packet, fs, snr_db=20, seed=0, key=KEY, **kwargs):
    # Как run_link, но принимает готовый packet (байты), а не текст.
    # Полезно, чтобы show_encoding и передача использовали ОДИН И ТОТ ЖЕ пакет
    # (при каждом pack_packet генерируется новый nonce — пакеты разные побитово).
    bits = bytes_to_bits(packet)
    cap = _capacity_bits(kind, fs, **kwargs)
    if cap is not None and len(bits) > cap:
        print(f'[{kind:>4}] передача невозможна: {len(bits)} бит > {cap}')
        return {'kind': kind, 'ok': False, 'reason': 'capacity'}

    tx = make_signal(kind, fs=fs, data_bits=bits, **kwargs)
    rx = apply_awgn(tx, snr_db, seed=seed)
    text_rx = receive(kind, rx, fs, len(bits), key=key, **kwargs)

    expected = unpack_packet(packet, key)
    ok = (text_rx is not None) and (text_rx == expected)
    print(f'[{kind:>4}] {"OK" if ok else "СБОЙ"}  (SNR={snr_db} дБ): {text_rx!r}')
    return {'kind': kind, 'ok': ok, 'text': text_rx, 'tx': tx}

# --- Показ процесса кодирования ---

def _fmt_bits(bits, max_show=128):
    # Красиво печатаем биты: если длинно — начало ... конец
    s = ''.join(str(int(b)) for b in bits)
    if len(s) <= max_show:
        return s
    half = max_show // 2
    return f'{s[:half]} ... {s[-half:]}'


def show_encoding(text, key=KEY):
    # Печатает полный конвейер: текст -> UTF-8 -> ChaCha20 -> пакет -> биты.
    # Возвращает (packet_bytes, bits), чтобы их можно было отдать в модулятор.
    print()
    print('=' * 78)
    print(' ЭТАП 1: КОДИРОВАНИЕ СООБЩЕНИЯ')
    print('=' * 78)

    # 1. Исходное
    print(f'1. Исходное сообщение        : {text!r}  ({len(text)} символов)')

    # 2. UTF-8
    raw = text.encode('utf-8')
    print(f'2. UTF-8 байты ({len(raw):>3} байт)     : {raw.hex(" ")}')

    # 3. Шифрование
    packet = pack_packet(text, key)
    body_len = int.from_bytes(packet[:HEADER_LEN], 'big')
    body = packet[HEADER_LEN:HEADER_LEN + body_len]
    nonce = body[:NONCE_LEN]
    ct = body[NONCE_LEN:]

    print(f'3. Шифрование ChaCha20-Poly1305:')
    print(f'   nonce ({len(nonce):>2} байт)            : {nonce.hex(" ")}')
    print(f'   ct+tag ({len(ct):>3} байт)           : {ct.hex(" ")}')
    print(f'     ├─ шифртекст {len(raw)} байт (как UTF-8, но зашифрован)')
    print(f'     └─ тег {TAG_LEN} байт (Poly1305, ловит любую ошибку)')

    # 4. Пакет
    print(f'4. Пакет ({len(packet):>3} байт)            : {packet.hex(" ")}')
    print(f'   └─ [len {HEADER_LEN}][nonce {NONCE_LEN}][ct+tag {len(ct)}]')

    # 5. Биты
    bits = bytes_to_bits(packet)
    print(f'5. Биты для модулятора ({len(bits)} бит)')
    print(f'   {_fmt_bits(bits)}')

    # 6. Контрольная расшифровка
    decoded = unpack_packet(packet, key)
    status = 'OK' if decoded == text else 'ОШИБКА'
    print(f'6. Контроль: расшифровка     : {decoded!r}  [{status}]')
    print()

    return packet, bits


def show_capacity(text, key=KEY):
    # Печатает, сколько бит нужно и влезает ли в каждый тип сигнала
    packet = pack_packet(text, key)
    n_bits = len(packet) * 8
    n_bytes = len(packet)

    print('=' * 78)
    print(f' ЭТАП 2: ЁМКОСТЬ КАНАЛОВ (нужно {n_bytes} байт = {n_bits} бит)')
    print('=' * 78)

    ofdm_bpsk = 512 * 8 * 1
    ofdm_qpsk = 512 * 8 * 2
    print(f'  OFDM BPSK (512×8×1)  : {ofdm_bpsk:>5} бит  '
          f'[{"OK " if n_bits <= ofdm_bpsk else "НЕТ"}]')
    print(f'  OFDM QPSK (512×8×2)  : {ofdm_qpsk:>5} бит  '
          f'[{"OK " if n_bits <= ofdm_qpsk else "НЕТ"}]')

    for n in (500, 1000, 2000):
        print(f'  LFM/HFM  n_bits={n:<5}: {n:>5} бит  '
              f'[{"OK " if n_bits <= n else "НЕТ"}]')

    print(f'  DSSS  sf=63          : без ограничения '
          f'(сигнал = {n_bits}×63 = {n_bits*63} отсчётов)')
    print()


def plot_ofdm_constellation(text, fs, snr_db=20,
                            N_fft=512, N_cp=64, N_sym=8, mod_order=4, key=KEY):
    # Созвездие OFDM: слева — что передано, справа — что принято.
    # Видно, что биты реально сидят в фазах поднесущих (4 точки = QPSK).
    packet = pack_packet(text, key)
    bits = bytes_to_bits(packet)
    tx = make_signal('ofdm', fs=fs, data_bits=bits,
                     N_fft=N_fft, N_cp=N_cp, N_sym=N_sym, mod_order=mod_order)
    rx = apply_awgn(tx, snr_db)

    sym_len = N_fft + N_cp
    qam_tx = np.fft.fft(tx[:sym_len*N_sym].reshape(N_sym, sym_len)[:, N_cp:], axis=1).ravel()
    qam_rx = np.fft.fft(rx[:sym_len*N_sym].reshape(N_sym, sym_len)[:, N_cp:], axis=1).ravel()

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].plot(qam_tx.real, qam_tx.imag, '.', ms=3, alpha=0.7)
    ax[0].set_title(f'Передано ({mod_order}-QAM) — {len(bits)} бит')
    ax[0].set_xlabel('I'); ax[0].set_ylabel('Q')
    ax[0].grid(alpha=0.3); ax[0].set_aspect('equal')

    ax[1].plot(qam_rx.real, qam_rx.imag, '.', ms=3, alpha=0.5, color='C1')
    ax[1].set_title(f'Принято (SNR = {snr_db} дБ)')
    ax[1].set_xlabel('I'); ax[1].set_ylabel('Q')
    ax[1].grid(alpha=0.3); ax[1].set_aspect('equal')

    plt.tight_layout()
    


def plot_dsss_correlation(text, fs, snr_db=20, sf=63, seed=0, key=KEY):
    # Корреляция чипов с PN: у передатчика — чёткие ±N, у приёмника — размытые шумом.
    # Видно, что биты «сжимаются» из широкополосного шума обратно в узкие отсчёты.
    packet = pack_packet(text, key)
    bits = bytes_to_bits(packet)
    tx = make_signal('dsss', fs=fs, data_bits=bits, sf=sf, seed=seed)
    rx = apply_awgn(tx, snr_db, seed=seed)

    n_sym = len(bits)
    rng = np.random.default_rng(seed)
    pn = rng.choice([-1, 1], size=sf)

    corr_tx = np.real(tx[:n_sym*sf].reshape(n_sym, sf) @ pn)
    corr_rx = np.real(rx[:n_sym*sf].reshape(n_sym, sf) @ pn)

    N_show = min(60, n_sym)
    idx = np.arange(N_show)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].stem(idx, corr_tx[:N_show], basefmt=' ')
    ax[0].set_title(f'Корреляция с PN — передано (первые {N_show} из {n_sym} бит)')
    ax[0].set_xlabel('Символ'); ax[0].set_ylabel('Амплитуда')
    ax[0].grid(alpha=0.3)

    ax[1].stem(idx, corr_rx[:N_show], basefmt=' ', linefmt='C1-', markerfmt='C1o')
    ax[1].set_title(f'Корреляция с PN — принято (SNR = {snr_db} дБ)')
    ax[1].set_xlabel('Символ'); ax[1].set_ylabel('Амплитуда')
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    # plt.show()


def plot_chirp_bpsk(text, fs, T=100e-3, f0=4e3, f1=16e3,
                    n_bits_chirp=1000, kind='lfm', key=KEY):
    # Три подграфика:
    #   1) Re(сигнал с данными)   — видно скачки знака
    #   2) Re(чистый чирп)        — гладкая огибающая
    #   3) |разность| / 2         — где сидят биты (плато 0/1)
    packet = pack_packet(text, key)
    bits = bytes_to_bits(packet)

    tx_data = make_signal(kind, fs=fs, data_bits=bits,
                          T=T, f0=f0, f1=f1, n_bits_chirp=n_bits_chirp)
    tx_clean = make_signal(kind, fs=fs, T=T, f0=f0, f1=f1)

    diff = np.abs(tx_data - tx_clean) / 2

    t = np.arange(len(tx_data)) / fs
    n_seg_show = min(8, n_bits_chirp)     # показываем первые 8 сегментов
    N_show = int(len(tx_data) * n_seg_show / n_bits_chirp)

    fig, ax = plt.subplots(3, 1, figsize=(11, 8))

    ax[0].plot(t[:N_show]*1e3, np.real(tx_data[:N_show]))
    ax[0].set_title(f'{kind.upper()} + BPSK: Re (первые {n_seg_show} сегментов)')
    ax[0].set_xlabel('t, мс'); ax[0].set_ylabel('Re')
    ax[0].grid(alpha=0.3)

    ax[1].plot(t[:N_show]*1e3, np.real(tx_clean[:N_show]))
    ax[1].set_title(f'{kind.upper()} без данных — чистый чирп (для сравнения)')
    ax[1].set_xlabel('t, мс'); ax[1].set_ylabel('Re')
    ax[1].grid(alpha=0.3)

    ax[2].plot(t[:N_show]*1e3, diff[:N_show])
    ax[2].set_title('|сигнал с данными − чистый| / 2 — вот где сидят биты (0 или 1)')
    ax[2].set_xlabel('t, мс'); ax[2].set_ylabel('бит')
    ax[2].set_ylim(-0.1, 1.1)
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    # plt.show()


def plot_ber_per_vs_snr(kind, text, fs, snr_list=None, seed=0, key=KEY, **kwargs):
    # BER (сколько бит пришло неверно) и PER (доля пакетов, где decrypt упал)
    # в зависимости от SNR. Это главный график работоспособности.
    if snr_list is None:
        snr_list = np.arange(-6, 22, 2)

    # Один фиксированный пакет — чтобы BER сравнивался корректно
    nonce = bytes(NONCE_LEN)
    ct = ChaCha20Poly1305(key).encrypt(nonce, text.encode('utf-8'), None)
    body = nonce + ct
    header = len(body).to_bytes(HEADER_LEN, 'big')
    packet = header + body
    bits_tx = bytes_to_bits(packet)

    ber_list, per_list = [], []
    for snr in snr_list:
        tx = make_signal(kind, fs=fs, data_bits=bits_tx, **kwargs)
        rx = apply_awgn(tx, snr, seed=seed)

        if kind == 'ofdm':
            bits_rx = demod_ofdm(rx, fs, kwargs['N_fft'], kwargs['N_cp'],
                                 kwargs['N_sym'], kwargs['mod_order'])
        elif kind == 'dsss':
            bits_rx = demod_dsss(rx, kwargs.get('sf', 63), seed=kwargs.get('seed', 0))
        else:
            bits_rx = demod_chirp(rx, fs, kwargs['T'], kwargs['f0'], kwargs['f1'],
                                  kind, kwargs['n_bits_chirp'])

        n = min(len(bits_tx), len(bits_rx))
        ber = float(np.mean(bits_tx[:n] != bits_rx[:n]))
        ber_list.append(max(ber, 1e-7))

        text_rx = receive(kind, rx, fs, len(bits_tx), key=key, **kwargs)
        per_list.append(0.0 if text_rx == text else 1.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].semilogy(snr_list, ber_list, 'o-')
    ax[0].set_xlabel('SNR, дБ'); ax[0].set_ylabel('BER')
    ax[0].set_title(f'{kind.upper()} — BER vs SNR')
    ax[0].grid(True, which='both', alpha=0.3)

    ax[1].plot(snr_list, per_list, 'o-', color='C1')
    ax[1].set_xlabel('SNR, дБ'); ax[1].set_ylabel('PER')
    ax[1].set_title(f'{kind.upper()} — PER vs SNR')
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].grid(True, alpha=0.3)