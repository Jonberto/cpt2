import tiktoken as tk
import struct as st

encd = tk.encoding_for_model('gpt2') # gets the encoding scheme for gpt2 wieghts depend on this encoding
f = open('data/tiny.txt', 'r')
tokens = encd.encode(f.read()) # encodes the text

with open('data/tokens', 'wb') as f_out:
    for t in tokens:
        value = st.pack("H", t)
        f_out.write(value)
    f_out.close()

structs = []
strings = b""
for i in range(50257):
    s = encd.decode_single_token_bytes(i)
    offset = len(strings)
    structs.append(offset)
    size = len(s)
    structs.append(size)
    strings += s

with open('data/enc', 'wb') as file_out:
    for e in structs:
        value = st.pack("I", e)
        file_out.write(value)
    print(len(strings))
    file_out.write(strings)


