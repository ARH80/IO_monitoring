import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys

from collections import Counter


class BLK:

    def __init__(self, directory):
        self.directory = directory
        self.df = None

        with open(F'{directory}/parsed_trace.txt', 'r') as f:
            self.lines = f.readlines()

        copy_list = []
        for line in self.lines:
            if line.startswith('CPU'):
                break

            copy_list.append(line)

        self.lines = copy_list
        last = self.lines[-1]
        last = float(last.split()[3])

        self.time_scope_size = int(last // 100)
        self.time_scope_size = max(self.time_scope_size, 1)
        self.address_scope_size = 25 * 1e6

    def get_features_dataframe(self):
        def parser(record):
            tokens = record.split()
            result = {
                'timestamp': float(tokens[3]),
                'time_scope': float(tokens[3]) // self.time_scope_size,
                'cid': int(tokens[1]),
                'sid': int(tokens[2]),
                'pid': int(tokens[4]),
                'action': tokens[5],
                'rw': tokens[6],
                'rw_spec': 'R' if 'R' in tokens[6] else 'W' if 'W' in tokens[6] else 'N',
                'start_address': int(tokens[7])
            }

            result.update({'n_sectors': int(tokens[9]) if '+' in tokens else 1})
            result.update({'size': result['n_sectors'] * 512})
            result.update({'end_address': result['start_address'] + result['size']})

            result.update({
                'scope_start_address': int(result['start_address'] // self.address_scope_size),
                'scope_end_address': int(result['start_address'] // self.address_scope_size)
            })

            return result

        df = pd.DataFrame([parser(x) for x in self.lines])
        self.df = df[df.rw_spec != 'N']

    def pie_plot(self):
        palette_color = sns.color_palette('bright')

        _ = plt.figure()
        plt.pie(
            [self.df[self.df.rw_spec == 'R'].shape[0], self.df[self.df.rw_spec == 'W'].shape[0]],
            labels=['Read', 'Write'],
            colors=palette_color,
            autopct='%.0f%%')

        plt.savefig(F'{self.directory}/pie')

    def density_on_size(self):
        _ = plt.figure()
        self.df.plot(kind='hist', column=['size'], by='rw_spec', bins=50)
        plt.savefig(F'{self.directory}/density_on_size.png')

    def rw_intensive_plot(self):
        def count_R(group):
            return group[group == 'R'].shape[0]

        def count_W(group):
            return group[group == 'W'].shape[0]

        df = self.df.groupby(['time_scope']).agg(
            r_count=('rw_spec', count_R), w_count=('rw_spec', count_W))

        _ = plt.figure()
        df.plot(kind='bar')
        plt.savefig(F'{self.directory}/rw_intensive.png')

    def address_frequency(self):
        c = Counter()
        _ = self.df[['scope_start_address', 'scope_end_address']].apply(
            lambda row: c.update(list(range(row[0], row[1] + 1))), axis=1)

        scopes, counts = zip(*c.items())

        _ = plt.figure()
        plt.bar(scopes, counts)
        plt.savefig(F'{self.directory}/address_frequency.png')


blk = BLK(sys.argv[1])
blk.get_features_dataframe()

blk.density_on_size()
blk.pie_plot()
blk.address_frequency()
blk.rw_intensive_plot()

plt.savefig('./blk-4main-plots.png')
