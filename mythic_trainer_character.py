"""
This script trains the  model

Usage:

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


settings = common.MythicSettings()


def random_training_set():
    inp = model.torch.LongTensor(settings.batch_size, settings.chunk_len)
    target = model.torch.LongTensor(settings.batch_size, settings.chunk_len)
    for bi in range(settings.batch_size):
        start_index = random.randint(0, settings.file_len - settings.chunk_len)
        end_index = start_index + settings.chunk_len + 1
        chunk = settings.file[start_index:end_index]
        inp[bi] = common.char_tensor(chunk[:-1])
        target[bi] = common.char_tensor(chunk[1:])
    inp = model.Variable(inp)
    target = model.Variable(target)
    if settings.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def train(input_pattern, target):
    hidden = decoder.init_hidden(settings.batch_size)
    if settings.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    this_loss = 0
    for c in range(settings.chunk_len):
        output, hidden = decoder(input_pattern[:, c], hidden)
        this_loss += criterion(output.view(settings.batch_size, -1), target[:, c])
    this_loss.backward()
    decoder_optimizer.step()
    return this_loss.data[0] / settings.chunk_len


def save():
    save_filename = os.path.splitext(os.path.basename(settings.filename))[0] + '.pt'
    model.torch.save(decoder, save_filename)
    log.out.info('Saved as %s' % save_filename)


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

    # Initialize models and start training
    decoder = model.CharRNN(
        common.num_characters,
        settings.hidden_size,
        common.num_characters,
        model=settings.model,
        n_layers=settings.layers,
    )
    decoder_optimizer = model.torch.optim.Adam(decoder.parameters(), lr=settings.learning_rate)
    criterion = model.nn.CrossEntropyLoss()

    if settings.cuda:
        decoder.cuda()

    all_losses = []
    loss_avg = 0
    try:
        print("Training for %d epochs..." % settings.epochs)
        for epoch in tqdm(range(1, settings.epochs + 1)):
            loss = train(*random_training_set())
            loss_avg += loss
            if epoch % settings.print_every == 0:
                all_losses.append(round(loss, 4))
                percent_done = epoch / settings.epochs * 100
                log.out.info('[%s (%d%%) %.4f]' % (common.time_since(start_time), percent_done, loss))
                log.out.info("\n" + writer.generate(decoder, 'Wh', 100, cuda=settings.cuda))
        log.out.info("Loss pattern:")
        log.out.info(",".join(map(str, all_losses)))
        log.out.info("Saving model.")
        save()

    except KeyboardInterrupt:
        print("Saving model before quit.")
        save()

    # Shut down and clean up
    total_time = round((time.time() - start_time), 0)
    log.out.info("Execution time: " + str(total_time) + " sec")
    log.out.info("All Done!")
    log.stopLog()
