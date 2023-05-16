import numpy as np
import help_functions as hf
import fock_class as fc

class edge:
    def __init__(self, angle, connected_leads, rates):
        self.angle  = angle
        self.cl     = np.array(connected_leads )
        self.rates  = np.array(rates )

        ### Implicitly defined attributes
        self.phase  = np.exp(1j*angle)
        self.tunnelings = hf.tunnel_from_gamma(self.rates)*self.phase
        self.majoranas  = []
        self.check_validity(print_pos_answer=False)

    def change_phase_angle(self, angle):
        self.angle  = angle
        self.phase  = np.exp(1j*angle)
        self.tunnelings = hf.tunnel_from_gamma(self.rates)*self.phase

    def check_validity(self, print_pos_answer=True):
        if self.rates.size == self.cl.size:
            if print_pos_answer:
                print('Lead connections are valid')
        else:
            print('Invalid lead connection, specify rate for each lead')

    def greet(self):
        print('Hi, I am an edge. I am connected to lead(s) {} with rate(s) {} and a phase angle of {:.3f}xPi. I contain {} Majorana(s).'.format(self.cl, self.rates, self.angle/np.pi, len(self.majoranas) ) )

    def create_majorana(self, index, wf_factor=1, wf_phase_angle=0, overlaps={}):
        majorana    = fc.maj_operator.from_edge(index, self, wf_factor=wf_factor, wf_phase_angle=wf_phase_angle, overlaps=overlaps)
        return majorana
