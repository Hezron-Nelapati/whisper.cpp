#!/usr/bin/env python3
# Convert an AI4Bharat IndicConformer model from NeMo format to ggml format.
#
# Usage: python convert-indicconformer-to-ggml.py --model model.nemo --language te --out-dir dir
#
# IndicConformer is a NeMo hybrid RNNT + CTC model with a ConformerEncoder
# (striding subsampling) and a multilingual head: one 257-way output layer per
# language, over a concatenated 22 x 256 vocabulary. This script picks a single
# language and writes a plain single-language RNNT model, so the runtime needs
# no notion of the other 21 languages:
#
#   vocabulary   -> the language's 256 token slice
#   joint head   -> joint.joint_net.2.<lang>  (256 tokens + blank at index 256)
#   embedding    -> rows 0..255 plus the all-zero SOS row, which becomes index 256
#
# The token fed back into the prediction network is the language-local id, which
# is what AI4Bharat's own inference code does.

import argparse
import io
import os
import struct
import sys
import tarfile

import numpy as np
import torch
import yaml

LSTM_PREFIX = 'decoder.prediction.dec_rnn.lstm'


def read_nemo(nemo_path):
    with tarfile.open(nemo_path, 'r') as tar:
        names = tar.getnames()

        def member(suffix):
            for n in names:
                if n.endswith(suffix):
                    return n
            raise FileNotFoundError(f'{suffix} not found in {nemo_path}')

        config = yaml.safe_load(tar.extractfile(member('model_config.yaml')).read())
        weights = io.BytesIO(tar.extractfile(member('model_weights.ckpt')).read())

    state_dict = torch.load(weights, map_location='cpu', weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    return config, state_dict


def write_tensor(fout, name, data, use_f16=True):
    if 'pre_encode.conv' in name and 'bias' in name and len(data.shape) == 1:
        data = data.reshape(1, -1, 1, 1)

    n_dims = len(data.shape)

    ftype = 1 if use_f16 else 0
    if use_f16:
        if n_dims < 2 or 'bias' in name or 'norm' in name or \
                ('pre_encode.conv' in name and n_dims == 4) or \
                'depthwise_conv.weight' in name:
            data = data.astype(np.float32)
            ftype = 0
        else:
            data = data.astype(np.float16)
    else:
        data = data.astype(np.float32)

    print(f'  {name} {list(data.shape)} {data.dtype}')
    name_bytes = name.encode('utf-8')
    fout.write(struct.pack('iii', n_dims, len(name_bytes), ftype))
    for i in range(n_dims):
        fout.write(struct.pack('i', data.shape[n_dims - 1 - i]))
    fout.write(name_bytes)
    data.tofile(fout)


def language_tensors(state_dict, config, lang):
    """Reduce the multilingual heads to a single language."""
    keys = config['joint']['language_keys']
    if lang not in keys:
        raise ValueError(f'language {lang!r} not in this model: {keys}')
    base = keys.index(lang) * 256

    vocab = config['joint']['vocabulary'][base:base + 256]

    head_w = state_dict[f'joint.joint_net.2.{lang}.weight']
    head_b = state_dict[f'joint.joint_net.2.{lang}.bias']
    if head_w.shape[0] != 257:
        raise ValueError(f'expected a 257-way head for {lang}, got {tuple(head_w.shape)}')

    # the prediction network is fed language-local ids; index 256 is the SOS/blank row
    embed = state_dict['decoder.prediction.embed.weight']
    embed = torch.cat([embed[:256], embed[-1:]], dim=0)

    return vocab, {
        'joint.joint_net.2.weight': head_w,
        'joint.joint_net.2.bias': head_b,
        'decoder.prediction.embed.weight': embed,
    }


def convert(nemo_path, lang, out_path, use_f16=True):
    config, state_dict = read_nemo(nemo_path)

    enc = config['encoder']
    if enc['subsampling'] != 'striding':
        raise ValueError(f"unsupported subsampling {enc['subsampling']!r}")
    if enc['self_attention_model'] != 'rel_pos':
        raise ValueError(f"unsupported attention {enc['self_attention_model']!r}")

    n_channels = enc['subsampling_conv_channels']
    if n_channels == -1:
        n_channels = enc['d_model']

    hparams = {
        'n_vocab': 256,
        'n_audio_ctx': 5000,
        'n_audio_state': enc['d_model'],
        'n_audio_head': enc['n_heads'],
        'n_audio_layer': enc['n_layers'],
        'n_mels': config['preprocessor']['features'],
        'n_fft': config['preprocessor']['n_fft'],
        'subsampling_factor': enc['subsampling_factor'],
        'n_subsampling_channels': n_channels,
        'n_conv_kernel': enc['conv_kernel_size'],
        'n_pred_dim': config['decoder']['prednet']['pred_hidden'],
        'n_pred_layers': config['decoder']['prednet']['pred_rnn_layers'],
        'n_tdt_durations': 0,           # plain RNN-T: this is what marks the arch
        'n_max_tokens': config['decoding']['greedy']['max_symbols'],
    }

    vocab, replaced = language_tensors(state_dict, config, lang)

    print('hyperparameters:')
    for k, v in hparams.items():
        print(f'  {k}: {v}')

    filters = state_dict['preprocessor.featurizer.fb'].squeeze().numpy().astype(np.float32)
    window = state_dict['preprocessor.featurizer.window'].squeeze().numpy().astype(np.float32)

    with open(out_path, 'wb') as fout:
        fout.write(struct.pack('i', 0x67676d6c))
        fout.write(struct.pack('i', hparams['n_vocab']))
        fout.write(struct.pack('i', hparams['n_audio_ctx']))
        fout.write(struct.pack('i', hparams['n_audio_state']))
        fout.write(struct.pack('i', hparams['n_audio_head']))
        fout.write(struct.pack('i', hparams['n_audio_layer']))
        fout.write(struct.pack('i', hparams['n_mels']))
        fout.write(struct.pack('i', 1 if use_f16 else 0))
        fout.write(struct.pack('i', hparams['n_fft']))
        fout.write(struct.pack('i', hparams['subsampling_factor']))
        fout.write(struct.pack('i', hparams['n_subsampling_channels']))
        fout.write(struct.pack('i', hparams['n_conv_kernel']))
        fout.write(struct.pack('i', hparams['n_pred_dim']))
        fout.write(struct.pack('i', hparams['n_pred_layers']))
        fout.write(struct.pack('i', hparams['n_tdt_durations']))
        fout.write(struct.pack('i', hparams['n_max_tokens']))

        fout.write(struct.pack('ii', filters.shape[0], filters.shape[1]))
        filters.tofile(fout)

        fout.write(struct.pack('i', window.shape[0]))
        window.tofile(fout)

        # no duration block: this model is a plain transducer

        fout.write(struct.pack('i', len(vocab)))
        for token in vocab:
            token_bytes = token.encode('utf-8')
            fout.write(struct.pack('i', len(token_bytes)))
            fout.write(token_bytes)

        bias_ih = {}
        for key, t in state_dict.items():
            if f'{LSTM_PREFIX}.bias_ih_l' in key:
                bias_ih[int(key.rsplit('bias_ih_l', 1)[1])] = t.squeeze().numpy().astype(np.float32)

        print('tensors:')
        for name, tensor in state_dict.items():
            if name.startswith('preprocessor.') or name.startswith('ctc_decoder.'):
                continue
            if name.startswith('joint.joint_net.2.'):
                continue                                    # written from `replaced` below
            if name == 'decoder.prediction.embed.weight':
                continue
            if f'{LSTM_PREFIX}.bias_ih_l' in name:
                continue                                    # folded into bias_hh

            if 'conv' in name and 'weight' in name and len(tensor.shape) == 4:
                data = tensor.numpy()
            else:
                data = tensor.squeeze().numpy()

            if name.startswith(f'{LSTM_PREFIX}.'):
                if f'{LSTM_PREFIX}.bias_hh_l' in name:
                    data = data.astype(np.float32) + bias_ih[int(name.rsplit('bias_hh_l', 1)[1])]
                    name = name.replace('bias_hh_l', 'bias_h_l')
                # pytorch gate order [i, f, g, o] -> [i, f, o, g], so the sigmoid gates are contiguous
                h = data.shape[0] // 4
                data = np.concatenate([data[:h], data[h:2 * h], data[3 * h:], data[2 * h:3 * h]], axis=0)

            write_tensor(fout, name, data, use_f16=use_f16)

        for name, tensor in replaced.items():
            write_tensor(fout, name, tensor.numpy(), use_f16=use_f16)

    size = os.path.getsize(out_path)
    print(f'\nwrote {out_path} ({size / (1024 ** 2):.2f} MB, {lang}, {len(vocab)} tokens)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert IndicConformer from NeMo to ggml')
    parser.add_argument('--model', required=True, help='path to the .nemo file')
    parser.add_argument('--language', required=True, help='language key to extract, e.g. te')
    parser.add_argument('--out-dir', required=True, help='directory to write the ggml model to')
    parser.add_argument('--out-name', default=None, help='output file name')
    parser.add_argument('--use-f32', action='store_true', help='keep weights at f32')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'error: {args.model} not found')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.out_name or f'ggml-indicconformer-{args.language}{"-f32" if args.use_f32 else ""}.bin'
    convert(args.model, args.language, os.path.join(args.out_dir, name), use_f16=not args.use_f32)
