type EncodeMessage = {
  type: 'encode';
  bitmap: ImageBitmap;
  width: number;
  height: number;
  quality: number;
};

type WorkerMessage = EncodeMessage;

let canvas: OffscreenCanvas | null = null;
let context: OffscreenCanvasRenderingContext2D | null = null;

const ensureCanvas = (width: number, height: number) => {
  if (!canvas || canvas.width !== width || canvas.height !== height) {
    canvas = new OffscreenCanvas(width, height);
    context = canvas.getContext('2d', { alpha: false });
  }
  return context;
};

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  if (message.type !== 'encode') {
    return;
  }

  const ctx = ensureCanvas(message.width, message.height);
  if (!ctx || !canvas) {
    message.bitmap.close();
    return;
  }

  ctx.drawImage(message.bitmap, 0, 0, message.width, message.height);
  message.bitmap.close();

  const blob = await canvas.convertToBlob({
    type: 'image/jpeg',
    quality: message.quality,
  });
  const buffer = await blob.arrayBuffer();

  self.postMessage(
    {
      type: 'frame',
      buffer,
      width: message.width,
      height: message.height,
    },
    [buffer],
  );
};
