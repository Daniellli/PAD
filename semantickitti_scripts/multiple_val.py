import argparse
import sys
import os
from datetime import datetime

def progressBar(i, max, text):

    bar_size = 60
    j = (i + 0) / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./multiple_remap.py")
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        default=None,
        help='.'
    )

    parser.add_argument(
        '--device',
        type=int,
        required=True,
        default=None,
        help='.'
    )

    parser.add_argument(
        '--start_epoch',
        type=int,
        required=False,
        default=0,
        help='.'
    )

    parser.add_argument(
        '--end_epoch',
        type=int,
        required=False,
        default=-1,
        help='.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.end_epoch == -1:
        FLAGS.end_epoch = FLAGS.start_epoch

    output_file_list = []
    command_list = []

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch + 1):

        output_dir = 'eval_outputs'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file_name = FLAGS.name + '.txt'
        output_path = output_dir + '/' + output_file_name

        progressBar(epoch - FLAGS.start_epoch, FLAGS.end_epoch + 1 - FLAGS.start_epoch, 'Generating Predictions ' + "(device=" + str(FLAGS.device) + ')')
        command = 'CUDA_VISIBLE_DEVICES=' + str(FLAGS.device) + ' python val_cylinder_asym_ood.py --load_path ' + "'../../semantic_kitti/checkpoints/" +  FLAGS.name + '/model_epoch_' + str(epoch) + ".pt'"

        if epoch == FLAGS.start_epoch:
            with open(output_path, 'w') as f:
                f.write("\n")


        with open(output_path, 'a') as f:
            f.write("\n" * 5)
            f.write('#' * 80 + "\n")
            now = datetime.now()
            f.write(str(now))
            f.write("\n")
            f.write(command + "\n")

        os.system(command + ' >>' + output_path + ' 2>&1')

        with open(output_path, 'a') as f:
            f.write("\n" + '#' * 80 + "\n")
            f.write("\n" * 5)

    progressBar(epoch - FLAGS.start_epoch + 1, FLAGS.end_epoch + 1 - FLAGS.start_epoch, 'Generating Predictions ' + "(device=" + str(FLAGS.device) + ')')
    print()
