# Distractor Aware Tracker (DAT)

This is the reimplementation of our generic object tracker as proposed in the
paper "In Defense of Color-based Model-free Tracking" by Horst Possegger et al.
This source-code is free for personal and academic use.
If you use it for a scientific publication, please cite:

        @InProceedings{ possegger15a, 
          author = {H. Possegger and T. Mauthner and H. Bischof}, 
          title = "{In Defense of Color-based Model-free Tracking}", 
          booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
          year = {2015}
        }

For further information on licensing, please have a look at the LICENCE.txt file.

## Get Started

Simply run `demo/demo.m` which demonstrates how to use the tracker interface on a provided image sequence.

This implementation has been tested under 64 bit Unix with Matlab R2011a, R2013a, and R2014a.


## VOT Integration

DAT can easily be integrated into the VOT framework for comparison.
Once you have set up the corresponding VOT workspace, you just have to modify the tracker command to point to the provided `dat_wrapper.m`.
In particular, the `tracker_DAT.m` script (automatically generated during VOT workspace initialization) should look like:

        tracker_label = 'DAT';
        tracker_command = generate_matlab_command('dat_wrapper', { '<path-to-dat>/src' });
        tracker_interpreter = 'matlab';


## Version History

  * v1.0 - Initial public release (2015 Aug 25).

