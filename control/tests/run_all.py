#!/usr/bin/env python
#
# test_all.py - test suit for python-control
# RMM, 30 Mar 2011

from __future__ import print_function
import unittest                 # unit test module
import re                       # regular expressions
import os                       # operating system commands

def test_all(verbosity=0):
    """ Runs all tests written for python-control.
    """
    try:                      # autodiscovery (python 2.7+)
        start_dir = './'
        pattern = '*_test.py'
        top_level_dir = '../'
        testModules = \
            unittest.defaultTestLoader.discover(start_dir, pattern=pattern, \
                                                top_level_dir=top_level_dir)

        for mod in test_mods:
            print('Running tests in', mod)
            tests = unittest.defaultTestLoader.loadTestFromModule(mod)
            t = unittest.TextTestRunner()
            t.run(tests)
            print('Completed tests in', mod)

    except:
        testModules = findTests('./tests/')

        # Now go through each module and run all of its tests.
        for mod in testModules:
            print('Running tests in', mod)
            suiteList=[]        # list of unittest.TestSuite objects
            exec('import '+mod+' as currentModule')

            try:
                currentSuite = currentModule.suite()
                if isinstance(currentSuite, unittest.TestSuite):
                    suiteList.append(currentModule.suite())
                else:
                    print(mod + '.suite() doesn\'t return a TestSuite')
            except:
                print('The test module '+mod+' doesnt have ' + \
                    'a proper suite() function')

            t=unittest.TextTestRunner(verbosity=verbosity)
            t.run(unittest.TestSuite(unittest.TestSuite(suiteList)))
            print('Completed tests in', mod)

def findTests(testdir = './', pattern = "[^.#]*_test.py$"):
    """Since python <2.7 doesn't have test discovery, this finds tests in the
    provided directory. The default is to check the current directory. Any files
    that match test* or Test* are considered unittest modules and checked for
    a module.suite() function (in tests())."""

    # Get list of files in test directory
    fileList = os.listdir(testdir)

    # Go through the files and look for anything that matches the pattern
    testModules= []
    for fileName in fileList:
        if (re.match(pattern, fileName)):
            testModules.append(fileName[:-len('.py')])

    # Return all of the modules that we find
    return testModules

if __name__=='__main__':
  test_all()
