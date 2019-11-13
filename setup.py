from distutils.core import setup
setup(
  name = 'migration_detector',         # How you named your package folder (MyLib)
  packages = ['migration_detector'],   # Chose the same as "name"
  version = '0.1.3',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A general approach to detect human migration events in digital trace data',   # Give a short description about your library
  author = 'Guanghua Chi',                   # Type in your name
  author_email = 'guanghua@berkeley.edu',      # Type in your E-Mail
  url = 'http://www.guanghuachi.com',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/g-chi/migration_detector/archive/v0.1.3.tar.gz',    # I explain this later on
  keywords = ['migration', 'trajectory', 'trace data'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'GraphLab-Create',
          'matplotlib',
          'seaborn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 2.7',      #Specify which pyhton versions that you want to support
  ],
)
