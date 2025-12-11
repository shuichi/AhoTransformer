import * as ort from 'onnxruntime-node';

// 語彙とトークナイザの定義（Python の DecoderTokenizer と同じ設定）
const maxLen = 7;
const padId = 0;
const sepId = 11;
const mask = 12;
const ahoId = 13;
const safeId = 14;
const digit2id = {};

for (let d = 0; d < 10; d++) {
  digit2id[String(d)] = d + 1;
}

function isAhoNumberJS(n) {
  const v = Math.abs(parseInt(n, 10));
  if (Number.isNaN(v)) return false;
  return v % 3 === 0 || v.toString().includes('3');
}

function encodeNumber(n) {
  const v = Math.abs(parseInt(n, 10));
  const s = v.toString();

  const digitIds = [];
  for (const ch of s) {
    digitIds.push(digit2id[ch]);
  }

  let tokens = digitIds.concat([sepId, labelId]);

  if (tokens.length > maxLen) {
    tokens = tokens.slice(tokens.length - maxLen);
  }

  const attentionMask = new Array(tokens.length).fill(1);
  while (tokens.length < maxLen) {
    tokens.push(padId);
    attentionMask.push(0);
  }

  return { inputIds: tokens, attentionMask: attentionMask };
}

let session = null;

async function initSession() {
  if (session) return session;

  try {
    // まず WebGPU での初期化を試みる
    console.log('Trying WebGPU...');
    session = await ort.InferenceSession.create('aho_decoder_transformer.onnx', {
      executionProviders: ['webgpu'],
    });
    console.log('ONNX session ready (WebGPU).');
  } catch (e) {
    // WebGPU でエラーが出たらここに来る
    console.warn('WebGPU failed or not supported. Falling back to WASM.', e);

    // WASM (CPU) で再挑戦
    session = await ort.InferenceSession.create('aho_decoder_transformer.onnx', {
      executionProviders: ['wasm'],
    });
    console.log('ONNX session ready (WASM).');
  }

  return session;
}

function logitsToResult(logitsTensor, attentionMask) {
  const data = logitsTensor.data; // Float32Array
  const dims = logitsTensor.dims; // 例: [1, 7, 15] など

  let B, S, V;

  if (dims.length === 3) {
    [B, S, V] = dims; // 期待パターン: [1, 7, 15]
  } else if (dims.length === 2) {
    // もし [S, V] になっている場合
    B = 1;
    [S, V] = dims;
  } else {
    console.error('予想外の logits 形状: ' + JSON.stringify(dims));
  }

  // PyTorch と同じく「mask=1 の最後の位置」をラベル位置とする
  let validLen = 0;
  for (let i = 0; i < S; i++) {
    if (attentionMask[i] === 1) validLen++;
  }
  const labelPos = validLen - 1;

  // data[0, labelPos, :] を取り出す
  // メモリレイアウト: index = (b*S + s)*V + v
  const b = 0;
  const offset = (b * S + labelPos) * V;
  const vec = [];
  for (let j = 0; j < V; j++) {
    vec.push(data[offset + j]);
  }

  // softmax
  const maxLogit = Math.max(...vec);
  const exps = vec.map((v) => Math.exp(v - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sumExp);

  const pAho = probs[ahoId];
  const pSafe = probs[safeId];

  let predId = 0;
  let predProb = -1;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > predProb) {
      predProb = probs[i];
      predId = i;
    }
  }

  let tag = '';
  if (predId === ahoId) {
    tag = 'Aho';
  } else if (predId === safeId) {
    tag = 'Safe';
  } else {
    tag = `その他(${predId})`;
  }

  return { tag, pAho, pSafe, predId };
}

let sum = 0;
const numbers = Array.from({ length: 100000 }, (_, i) => i);
for (const n of numbers) {
  // ONNX Runtime Web 用の Tensor を作成
  // int64 は BigInt64Array を使う
  const { inputIds, attentionMask } = encodeNumber(n);
  const inputIdsBig = new BigInt64Array(inputIds.length);
  const attnMaskBig = new BigInt64Array(attentionMask.length);

  for (let i = 0; i < inputIds.length; i++) {
    inputIdsBig[i] = BigInt(inputIds[i]);
    attnMaskBig[i] = BigInt(attentionMask[i]);
  }

  const inputTensor = new ort.Tensor('int64', inputIdsBig, [1, maxLen]);
  const attnTensor = new ort.Tensor('int64', attnMaskBig, [1, maxLen]);

  const sess = await initSession();

  const feeds = {
    input_ids: inputTensor,
    attention_mask: attnTensor,
  };

  const outputs = await sess.run(feeds);
  const logitsTensor = outputs.logits; // Tensor オブジェクト
  const { tag, pAho, pSafe } = logitsToResult(logitsTensor, attentionMask);

  const rule = isAhoNumberJS(n); // 検証用途に JS で正解を計算
  const result = rule === (tag === 'Aho') || rule !== (tag === 'Safe');
  if (result) sum++;
  const text = `n = ${n}  推論結果: ${tag}   p_AHO = ${pAho.toFixed(3)}   p_SAFE = ${pSafe.toFixed(3)}   ${
    result ? 'OK' : 'NG'
  }`;
  console.log(text);
}

console.log(`正解数: ${sum} / ${numbers.length}`);
