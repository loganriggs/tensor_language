"""Byte-aware prefix/next-token labels for descriptive continuation tests."""
import codecs

def annotate(row,encoding):
    decoder=codecs.getincrementaldecoder('utf-8')('replace');decoded='';result=[]
    for position,token in enumerate(row):
        decoded+=decoder.decode(encoding.decode_single_token_bytes(int(token)),final=False)
        pending=bool(decoder.getstate()[0]);last=decoded[-1:] if decoded else ''
        if position+1<len(row):
            nxt=encoding.decode_single_token_bytes(int(row[position+1]));trim=nxt.lstrip()
            letter=lambda x:bool(x) and (65<=x[0]<=90 or 97<=x[0]<=122)
            eligible=not pending and last.isalpha()
            result.append(dict(utf8_pending=pending,eligible=eligible,continuation=eligible and letter(nxt),spaced_word=eligible and bool(nxt[:1].isspace()) and letter(trim)))
    return result

def vocabulary_masks(encoding,width):
    bare=[];spaced=[]
    for i in range(width):
        if i>=encoding.n_vocab:continue
        b=encoding.decode_single_token_bytes(i);trim=b.lstrip()
        if b and (65<=b[0]<=90 or 97<=b[0]<=122):bare.append(i)
        if b[:1].isspace() and trim and (65<=trim[0]<=90 or 97<=trim[0]<=122):spaced.append(i)
    return bare,spaced
