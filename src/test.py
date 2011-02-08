import subprocess as SP

def tests_old():
  """ Runs all of the tests written for python-control. This should be 
  changed in the future so it does run seperate main functions/scripts,
  but is integrated into the package. Also, the tests should be in their
  own directory /trunk/tests. Running the test should be as simple as:
  "import control; control.tests()"
  """
  testList = ['TestBDAlg.py','TestConvert.py','TestFreqRsp.py',\
             'TestMatlab.py','TestModelsimp.py','TestSlycot.py',\
             'TestStateSp.py','TestStatefbk.py','TestXferFcn.py']
  #Add more tests to this list as they are created. Each is assumed to run
  #as a script, as is usually done with unittest.
  for test in testList:
    print 'Running',test  
    print SP.Popen(['./'+test],stdout=SP.PIPE).communicate()[0]
    print 'Completed',test


###############################################################################

def getFileList(dir,fileExtension=''):
  """ Finds all files in the given directory that have the given file extension"""
  filesRaw = SP.Popen(['ls',dir],stdout=SP.PIPE).communicate()[0]
  #files separated by endlines
  filename= ''
  fileList=[]
  #print 'filesRaw is ',filesRaw
  for c in filesRaw:
    if c!='\n':
      filename+=c
    else: #completed file name
      if fileExtension != '' and filename[-len(fileExtension):] == fileExtension:
        fileList.append(filename)
      else:
        pass #fileList.append(dir+filename)
      filename=''
  return fileList

###############################################################################


def findTests(testdir='./'):
  """Since python <2.7 doesn't have test discovery, this finds tests in the 
  provided directory. The default is to check the current directory. Any files
  that match test* or Test* are considered unittest modules and checked for 
  a module.suite() function (in tests())."""
  fileList = getFileList(testdir,fileExtension='.py')
  testModules= []
  for fileName in fileList:
    if (fileName[:4] =='test' or fileName[:4]=='Test') and fileName!='test.py':
       testModules.append(fileName[:-len('.py')])
  return testModules
  
###############################################################################
  
  
def tests():
  import unittest
  try: #auto test discovery is only implemented in python 2.7+
    start_dir='./' #change to a tests directory eventually.
    pattern = 'Test*.py'
    top_level_dir = './' #this might change? see 
    #http://docs.python.org/library/unittest.html#unittest.TestLoader.discover
    test_mods=unittest.defaultTestLoader.discover(start_dir,pattern=pattern,\
      top_level_dir=top_level_dir)
    #now go through each module and run all of its tests.
    print 'found test mods and they are',test_mods
    for mod in test_mods:
      print 'Running tests in',mod
      tests = unittest.defaultTestLoader.loadTestFromModule(mod)
      t = unittest.TextTestRunner()
      t.run(tests)
      print 'Completed tests in',mod
  except: 
    #If can't do auto discovery, for now it is hard-coded. This is not ideal for
    #when new tests are added or existing ones are reorganized/renamed.
    
    #remove all of the print commands once tests are debugged and converted to 
    #unittests.
    
    print 'Tests may be incomplete'
    t=unittest.TextTestRunner()

    testModules = findTests()

    suite = unittest.TestSuite() 
    suiteList=[]
    for mod in testModules:
      exec('import '+mod+' as currentModule')
      #After tests have been debugged and made into unittests, remove 
      #the above (except the import) and replace with something like this:
      try:
        currentSuite = currentModule.suite()
        if isinstance(currentSuite,unittest.TestSuite):
          suiteList.append(currentModule.suite())
        else:
          print mod+'.suite() doesnt return a unittest.TestSuite!, please fix!'
      except:
        print 'The test module '+mod+' doesnt have '+\
          'a proper suite() function that returns a unittest.TestSuite object'+\
            ' Please fix this!'
    alltests = unittest.TestSuite(suiteList)
    t.run(unittest.TestSuite(alltests)) 

###############################################################################
    
if __name__=='__main__':
  tests()
