# Character-level RNN

A PyTorch implementation of [char-rnn](https://github.com/karpathy/char-rnn) for character-level text generation. This is copied from [the Practical PyTorch series](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb).

## Training

On GPU box, run `python mythic_trainer_character.py --cuda --text_file tinyshakespeare.txt`; omit `--cuda` if you don't have a GPU.