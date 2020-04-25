# pyhw

# PyHW

This is a 2D pseudo-spectral Hasegawa-Wakatani solver, written mostly in python, where the number crunching is done on C using ctypes.


Quick start
-----------

Go to a directory where you want to put the source code and clone the repository:

    $ git clone https://github.com/gurcani/pyhw.git

Then you can change into the generated directory 'pyhw' and install the python code locally by running (notice the dot at the end):

    $ pip install --user .
    
This will install pyhw as a package into your local user python distribution, usually some directory like '~/.local/lib/python3.8/site-packages/'. Once the package is installed you can delete the source directory if you want.

now on ipython, or spyder, you can do:

```python

In [1]: import pyhw.sim as phs

In [2]: phs.pars['C']=0.01

In [3]: phs.pars['kap']=0.5

In [4]: phs.pars['nu']=1e-3

In [5]: phs.pars['D']=1e-3

In [6]: phs.set_res(1024,1024)

In [7]: phs.run('out.h5')
t= 0.0
0.05952000617980957 seconds elapsed.
t= 1.0000000000000007
45.792287826538086 seconds elapsed.
t= 2.0000000000000013
92.95970797538757 seconds elapsed.

```

By default the dissipation terms are standard viscosity  &nu; k<sup>2</sup> and D k<sup>2</sup>. You can swithc to hyper viscosity by calling the run function as follows:

```python
phs.run('out.h5',nuf = lambda nu,ksqr : nu*ksqr**2)
```

One can also use the pure_python implementation of the same code, which is about two times slowed than the ctypes version.

```python
phs.run('out.h5',pure_python=True)
```

There is also a number of utilities to be used along with pyhw. They can be used both locally or remotely if both local and remote systems are nicely configured. Notice that the hdf5 file 'out.h5' generated by the code is in general of the order of several tens of gigabytes. Therefore copying the whole file or remotely accessing it is time consuming. In order to remedy that, the code also writes a preview file which is named 'out_prev.h5' if the original file is 'out.h5'. This preview file contains only the last 10 time steps of the run, and therefore is generally only a few hundred megabytes. 

Here is an example of the remote usage:

```python
In [1]: import pyhw.utils as phu

In [2]: phu.plot_spec('remote.polytechnique.fr:~/pyprj/pyhwruns/run1/out_pyhw_prev.h5')
```

here I have a remote machine called 'remote.polytechnique.fr' in which I have an ongoing simulation running at the directory ~/pyprj/pyhwruns/run1/ . Notice that I gave the name of the preview file as an argument. Since the file is copied over into the local directory, it is better that it is only 78.16MB.

The utilities can also be used to generate various animations. For example:

```python
phu.spec_anim('remote.polytechnique.fr:~/pyprj/pyhwruns/run1/out_pyhw.h5','out.mp4',vmin=1e-12,vmax=1e-2)
```

will download the whole file, and generate an animated spectrum where the y range of the shown spectrum will be limited to vmin and vmax values that are given. The animation will be saved as 'out.mp4' and vlc will be launched to show that file. The vmin and vmax are optional but usually better to set by hand.

Note that once the transfer is done, if you want to call one of utility functions like plot_spec, or anim, you can use the local file 'pyhw_tempdir/out_pyhw.h5' as its argument instead of the remote file 'remote.polytechnique.fr:~/pyprj/pyhwruns/run1/out_pyhw.h5'.
