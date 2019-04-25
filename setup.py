from setuptools import setup

setup(name='edpy',
      version='0.1',
      description='Python Utils for configure/analyze ED2 model',
      url='https://github.com/xiangtaoxu/edpy.git',
      author='Xiangtao Xu',
      author_email='xu.withoutwax@gmail.com',
      license='MIT',
      packages=['edpy'],
      install_requires=['numpy',
                        'pandas',
                        'lxml',
                        'h5py',
                        'matplotlib',
                        'fpdf']
      zip_safe=False)
