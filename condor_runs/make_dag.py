#! /usr/bin/env python
import sys

def add_job(the_file, job_number, **kwargs):

    job_id="%s%.6u" % ('GW', job_number)
    the_file.write("JOB %s %s.sub\n" % (job_id, 'GW'))
    vars_line=" ".join(['%s="%s"'%(arg,str(val))
                        for arg,val in kwargs.iteritems()])
    the_file.write("VARS %s %s\n" % (job_id, vars_line))
    the_file.write("\n")

if __name__ == "__main__":

    r = 5
    test_samples=int(r*r)

    fdag = open("my.dag",'w')
    for idx in range(test_samples):
        add_job(fdag, idx, label='test_samp_%d' % idx, od='bilby_output')
