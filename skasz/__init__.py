import sys
import os

splist = sys.path
splist = [s for s in splist if 'site-packages' in s and os.path.exists(s+'/rascil')][0]

RASCIL_DATA = '{0}/rascil/data'.format(splist)
os.environ['RASCIL_DATA'] = RASCIL_DATA

def download_rascil_data():
    os.system('curl https://ska-telescope.gitlab.io/external/rascil-main/rascil_data.tgz -o rascil_data.tgz')
    os.system('tar -xzf rascil_data.tgz')
    os.system('mv data {0}'.format(RASCIL_DATA))
    os.system('rm -vf rascil_data.tgz')


if not os.path.exists(RASCIL_DATA):
    getrascildata = input('Would you like to download the RASCIL data? [yes]/no ').strip().lower()
    getrascildata = True if getrascildata in ['','yes','y'] else False

    if getrascildata: 
        try: download_rascil_data()
        except: print('Failed to download RASCIL data. You can try again running skssz.download_rascil_data().')
    else: print('RASCIL data not downloaded. Some simulations may fail.')
        