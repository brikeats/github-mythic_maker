"""
This script trains a character level RNN (LSTM) model

Usage:
python mythic_trainer_character.py --text_file ./DATA/foo.txt --print_every 100 --epochs 2000


@author: Brad Beechler (brad.e.beechler@gmail.com)
# Last Modification: 09/20/2017 (Brad Beechler)
"""

from uplog import log
import os
import time
import random
from tqdm import tqdm
import mythic_common as common
import mythic_writer_character as writer
import mythic_model_character as model

# This is hamfisted but these setting are used so much it's convienient
settings = common.TrainerSettings()


def random_training_set():
    inp = model.torch.LongTensor(settings.batch_size, settings.chunk_size)
    target = model.torch.LongTensor(settings.batch_size, settings.chunk_size)
    for bi in range(settings.batch_size):
        start_index = random.randint(0, settings.text_length - settings.chunk_size)
        end_index = start_index + settings.chunk_size + 1
        chunk = settings.text_string[start_index:end_index]
        inp[bi] = common.char_tensor(chunk[:-1])
        target[bi] = common.char_tensor(chunk[1:])
    inp = model.Variable(inp)
    target = model.Variable(target)
    if settings.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def ordered_training_set(start_index):
    if start_index > (settings.text_length - settings.chunk_size):
        log.out.warning("Requested index would blow bounds in text array, setting to random.")
        start_index = random.randint(0, settings.text_length - settings.chunk_size)
    inp = model.torch.LongTensor(settings.batch_size, settings.chunk_size)
    target = model.torch.LongTensor(settings.batch_size, settings.chunk_size)
    for bi in range(settings.batch_size):
        # start_index = random.randint(0, settings.text_length - settings.chunk_size)
        end_index = start_index + settings.chunk_size + 1
        chunk = settings.text_string[start_index:end_index]
        inp[bi] = common.char_tensor(chunk[:-1])
        target[bi] = common.char_tensor(chunk[1:])
    inp = model.Variable(inp)
    target = model.Variable(target)
    if settings.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target, end_index


def train(input_pattern, target):
    hidden = decoder.init_hidden(settings.batch_size)
    if settings.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    this_loss = 0
    for c in range(settings.chunk_size):
        output, hidden = decoder(input_pattern[:, c], hidden)
        this_loss += criterion(output.view(settings.batch_size, -1), target[:, c])
    this_loss.backward()
    decoder_optimizer.step()
    return this_loss.data[0] / settings.chunk_size


def save(save_filename):
    model.torch.save(decoder, save_filename)
    log.out.info("Saved as:" + save_filename)


if __name__ == '__main__':
    """
    Main driver.
    """
    # Set the logging level to normal and start run
    start_time = time.time()
    common.report_sys_info()
    if settings.debug:
        log.out.setLevel('DEBUG')
    else:
        log.out.setLevel('INFO')
    if settings.cuda:
        log.out.info("Using CUDA")
    else:
        log.out.info("Using CPU")
    if settings.model_file is None:
        save_filename = os.path.splitext(os.path.basename(settings.text_file))[0] + '.pt'
    else:
        save_filename = settings.model_file
    settings.report()

    log.out.info("Read text data from: " + settings.text_file)
    log.out.info("Found " + str(settings.text_length) + " characters.")
    # Initialize models and start training
    decoder = model.CharRNN(
        common.num_characters,
        settings.hidden_size,
        common.num_characters,
        model=settings.model,
        n_layers=settings.layers,
    )
    # Set the optimizer
    decoder_optimizer = model.torch.optim.Adam(decoder.parameters(), lr=settings.learning_rate)
    # Set the loss function (criterion)
    criterion = model.nn.CrossEntropyLoss()

    if settings.cuda:
        decoder.cuda()

    all_losses = []
    loss_avg = 0
    try:
        for epoch in tqdm(range(1, settings.epochs + 1)):
            loss = train(*random_training_set())
            loss_avg += loss
            if epoch % settings.print_every == 0:
                all_losses.append(round(loss, 4))
                percent_done = epoch / settings.epochs * 100
                log.out.info('[%s (%d%%) %.4f]' % (common.time_since(start_time), percent_done, loss))
                log.out.info("\n" + writer.generate(decoder, 'Wh', 100, cuda=settings.cuda))
        log.out.info("Loss history:")
        log.out.info(",".join(map(str, all_losses)))
        log.out.info("Saving model.")
        save('model_checkpoint.dat')

    except KeyboardInterrupt:
        log.out.info("Saving model before quit.")
        save('model_checkpoint.dat')

    # Shut down and clean up
    total_time = round((time.time() - start_time), 0)
    log.out.info("Execution time: " + str(total_time) + " sec")
    log.out.info("All Done!")
    log.stopLog()
