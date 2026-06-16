# Qwen3-VL Demo Narration Script

Total runtime: 02:45

## Source metrics

- Qwen3-VL-8B word-level: CER 0.035, word IoU 0.722, latency 26.8s.
- Google Document AI word-level: CER 0.108, word IoU 0.611, latency 3.6s.
- Evidence source: existing benchmark JSON and visualization PNG artifacts only.
- No live model run, no cloud API call, no generated voice audio.

## Timed narration

### 00:00-00:20 - Qwen3-VL for handwriting OCR + localization

This demo shows Qwen3-VL on the handwriting problem that matters for essay feedback: transcription plus word-level localization. A useful system needs both the text and the boxes.

On-screen caption: The task is not just reading handwriting. The feedback pipeline also needs word locations, so every detected error can point back to the page.

### 00:20-00:45 - Evaluation setup: crop away the easy text

The benchmark avoids a common trap. IAM evaluation forms include a machine-printed prompt and a handwritten copy of the same text. These numbers come from cropped handwritten regions only.

On-screen caption: Full IAM forms contain the same text twice. The benchmark uses only the handwritten crop, so the model cannot win by reading the printed prompt.

### 00:45-01:25 - Main result: lower CER and stronger word boxes

On the 25-image handwritten benchmark, Qwen3-VL-8B is the best automatable result: zero point zero three five CER and zero point seven two two word IoU. Google Document AI is at zero point one zero eight CER and zero point six one one word IoU.

On-screen caption: Qwen3-VL-8B reaches CER 0.035 and word IoU 0.722 on 25 handwritten crops, beating the Google Document AI baseline on both metrics.

### 01:25-01:55 - Visual proof: predicted boxes line up with words

Here is the same handwritten page with word boxes overlaid. Green is ground truth and red is prediction. The point is not a pretty overlay; it is auditable localization for later feedback.

On-screen caption: Green is ground truth. Red is the model prediction. Qwen's red boxes track the handwritten words tightly enough for downstream error feedback.

### 01:55-02:20 - Honest range: best, median, and worst examples

The benchmark is not cherry-picked. This section shows a clean page, a median page, and the highest-error Qwen3-VL-8B example so the range of behavior is visible.

On-screen caption: The result is strong, but not magic. The demo includes a best case, a median page, and the highest-CER Qwen3-VL-8B sample.

### 02:20-02:45 - Practical conclusion

The practical conclusion is that Qwen3-VL is the only evaluated candidate that gives strong handwriting transcription and usable word boxes. The four billion parameter model runs locally; the eight billion result here was evaluated through an API because local INT4 is blocked on this Blackwell setup.

On-screen caption: Qwen3-VL is the only evaluated approach that combines strong handwriting transcription with usable word localization.
