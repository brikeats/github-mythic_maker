"""
This script writes out prediction given a trained model

Usage:
python mythic_writer_character.py --model_file ./foo.pt --seed_string Bar

@author: Brad Beechler (brad.e.beechler@gmail.com)
# Last Modification: 09/20/2017 (Brad Beechler)
"""
from uplog import log
import torch
import mythic_common as common
import mythic_model_character


def generate(decoder, seed_string='A', predict_length=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = mythic_model_character.Variable(common.char_tensor(seed_string).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = seed_string

    # Use priming string to "build up" hidden state
    for p in range(len(seed_string) - 1):
        _, hidden = decoder(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_length):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = common.trainable_characters[top_i]
        predicted += predicted_char
        inp = mythic_model_character.Variable(common.char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted


if __name__ == '__main__':
    """
    CLI driver to rerun a model prediction
    """
    # Get settings from command line
    write_settings = common.WriterSettings()
    if write_settings.debug:
        log.out.setLevel('DEBUG')
    else:
        log.out.setLevel('INFO')

    # Parse command line arguments
    log.out.info("Loading model from file: " + write_settings.model_file)
    decoder = torch.load(write_settings.model_file)
    predicted_string = generate(decoder,
                                seed_string=write_settings.seed_string,
                                predict_length=write_settings.predict_length,
                                temperature=write_settings.temperature,
                                cuda=write_settings.cuda)
    log.out.info("Seed string: " + "\n" + write_settings.seed_string)
    log.out.info("Predicted string: " + "\n" + predicted_string)
