/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // initialize ParticleFilter
    num_particles = 40;
    particles.resize(num_particles);
    weights.resize(num_particles);

    //init particles with normal distribution
    double init_weight = 1.0;

    normal_distribution<double> dist_x(x, std[0]), dist_y(y, std[1]), dist_theta(theta, std[2]);
    default_random_engine gen;
    for(int i=0; i< num_particles; i++){
        particles[i] = {i, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight};
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // using the CTRV model
    // predict state using time, gps uncertainty, vect1 and vect2

    for (int i=0; i< num_particles; i++){
    // if yaw rate is 0
        if (fabs(yaw_rate) < 0.001) {
            particles[i].x += velocity * cos(particles[i].theta) * delta_t;
            particles[i].y += velocity * sin(particles[i].theta) * delta_t;
        }
        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t)
                                                     - sin(particles[i].theta));

            particles[i].y += velocity / yaw_rate * (-cos(particles[i].theta + yaw_rate * delta_t)
                                                     + cos(particles[i].theta));
        }
        particles[i].theta = particles[i].theta + (yaw_rate * delta_t);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); i++)
    {
        double min_dist = 50000;
        int min_index = -1;
        for (int j = 0; j < predicted.size(); j++)
        {
            // calculate dist
            double dist = sqrt(pow(observations[i].x - predicted[j].x, 2) +
                               pow(observations[i].y - predicted[j].y, 2));

            if (min_dist > dist)
            {
                min_dist = dist;
                min_index = j;
            }
        }

        if (min_index != -1)
            observations[i].id = predicted[min_index].id;
        else
            cout << "\n dataAssociation Error!!!";
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    // get predicted landmarks
    std::vector<LandmarkObs> pred_landmarks(num_particles);
    _get_pred_landmarks(pred_landmarks, map_landmarks);
    for (int i = 0; i < num_particles; i++)
    {
        // 1. change coords
        std::vector<LandmarkObs> meas_landmarks(observations.size());
        _to_map_coord(meas_landmarks, observations, particles[i]);

        // 2. nearest landmarks
        dataAssociation(pred_landmarks, meas_landmarks);

        // 3. update weights
        particles[i].weight = _calc_weight(pred_landmarks, meas_landmarks, std_landmark[0], std_landmark[1]);
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // resample particles 
    default_random_engine gen;
    vector<Particle> resampled_particles(num_particles);
    vector<double> weights(num_particles);

    // Get particles weights
    for (int i = 0; i < num_particles; i++) {
        weights[i] = particles[i].weight;
    }

    discrete_distribution<int> index(weights.begin(), weights.end());
    for (unsigned j=0; j<num_particles;j++){
        int i = index(gen);
        resampled_particles[j] = particles[i];
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
