import logging
from cladeomatic.constants import LOG_FORMAT
from scipy.stats import entropy,fisher_exact
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from subprocess import Popen, PIPE

def init_console_logger(lvl):
    """

    Parameters
    ----------
    lvl [int] : Integer of level of logging desired 0,1,2,3

    Returns
    -------

    logging object

    """
    logging_levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    report_lvl = logging_levels[lvl]

    logging.basicConfig(format=LOG_FORMAT, level=report_lvl)
    return logging


def run_command(command):
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')
    return stdout, stderr

def calc_shanon_entropy(value_list):
    total = sum(value_list)
    if total == 0:
        return -1
    values = []
    for v in value_list:
        values.append(v / total)
    return entropy(values)

def calc_AMI(category_1, category_2):
    return adjusted_mutual_info_score(category_1, category_2, average_method='arithmetic')

def calc_ARI(category_1, category_2):
    return adjusted_rand_score(category_1, category_2)
