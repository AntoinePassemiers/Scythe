/**
    exceptions.hpp
    Custom exceptions
    
    @author Antoine Passemiers
    @version 1.0 24/06/2017
*/

#ifndef EXCEPTIONS_HPP_
#define EXCEPTIONS_HPP_

#include <stdexcept>


namespace scythe {

class OOPException : public std::runtime_error {
public:
    // When an overriden method must never be called
    // Error in the object-oriented design of the class
    OOPException() : runtime_error("Object-oriented programming error"){}
    OOPException(std::string msg) : runtime_error(msg.c_str()) {}
};

class WrongVirtualDatasetException : public std::exception {
public:
    const char* what() const throw() { return "This kind of virtual dataset is not compatible\n"; }
};

class UnhandledDtypeException : public std::exception {
public:
    const char* what() const throw() { return "Unknown primitive data type.\n"; }
};

} // namespace

#endif // EXCEPTIONS_HPP_