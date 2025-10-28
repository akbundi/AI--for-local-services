import React, { useState, useContext, useEffect } from 'react';
import { AuthContext, API } from '@/App';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Calendar } from '@/components/ui/calendar';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import axios from 'axios';
import { MapPin, Search, Star, Phone, Globe, DollarSign, Clock, MessageSquare, X, Calendar as CalendarIcon, LogOut, BookOpen, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { format } from 'date-fns';

const Dashboard = () => {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();
  
  const [location, setLocation] = useState('');
  const [serviceType, setServiceType] = useState('');
  const [budget, setBudget] = useState('');
  const [providers, setProviders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [showBooking, setShowBooking] = useState(false);
  const [bookingDate, setBookingDate] = useState(new Date());
  const [bookingTime, setBookingTime] = useState('');
  const [bookingNotes, setBookingNotes] = useState('');
  
  // Chat state
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  const handleSearch = async () => {
    if (!location || !serviceType) {
      toast.error('Please enter location and select service type');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/services/search`, {
        location,
        service_type: serviceType,
        radius: 5000,
        budget: budget || null
      });
      setProviders(response.data.providers);
      toast.success(`Found ${response.data.total} providers`);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleBooking = async () => {
    if (!selectedProvider || !bookingTime) {
      toast.error('Please select date and time');
      return;
    }

    try {
      await axios.post(`${API}/bookings/create`, {
        provider_name: selectedProvider.name,
        provider_address: selectedProvider.address,
        service_type: serviceType,
        booking_date: format(bookingDate, 'yyyy-MM-dd'),
        booking_time: bookingTime,
        notes: bookingNotes
      });
      toast.success('Booking created successfully!');
      setShowBooking(false);
      setSelectedProvider(null);
      setBookingNotes('');
      setBookingTime('');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Booking failed');
    }
  };

  const handleChatSend = async () => {
    if (!chatInput.trim()) return;

    const userMessage = chatInput;
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await axios.post(`${API}/chat/send`, {
        message: userMessage,
        location: location || null
      });
      setChatMessages(prev => [...prev, { role: 'assistant', content: response.data.response }]);
    } catch (error) {
      toast.error('Chat failed');
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50" data-testid="dashboard">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <MapPin className="w-6 h-6 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">LocalProHelper</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">Welcome, {user?.username}!</span>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => navigate('/bookings')}
              data-testid="my-bookings-btn"
            >
              <BookOpen className="w-4 h-4 mr-2" />
              My Bookings
            </Button>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={logout}
              data-testid="logout-btn"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Search Section */}
        <Card className="mb-8 shadow-lg" data-testid="search-section">
          <CardHeader>
            <CardTitle className="text-2xl">Find Local Service Providers</CardTitle>
            <CardDescription>Enter your location and select the service you need</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-4 gap-4">
              <div className="md:col-span-2">
                <Label htmlFor="location">Location</Label>
                <Input 
                  id="location" 
                  placeholder="Enter city or address" 
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  data-testid="location-input"
                />
              </div>
              <div>
                <Label htmlFor="service">Service Type</Label>
                <Select value={serviceType} onValueChange={setServiceType}>
                  <SelectTrigger data-testid="service-type-select">
                    <SelectValue placeholder="Select service" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="plumber">Plumber</SelectItem>
                    <SelectItem value="tutor">Tutor</SelectItem>
                    <SelectItem value="gym">Gym</SelectItem>
                    <SelectItem value="repair">Repair Professional</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="budget">Budget</Label>
                <Select value={budget} onValueChange={setBudget}>
                  <SelectTrigger data-testid="budget-select">
                    <SelectValue placeholder="Any budget" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cheap">Cheap ($)</SelectItem>
                    <SelectItem value="mediocre">Mediocre ($$)</SelectItem>
                    <SelectItem value="premium">Premium ($$$)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="mt-6 flex gap-4">
              <Button 
                onClick={handleSearch} 
                disabled={loading} 
                className="bg-blue-600 hover:bg-blue-700"
                data-testid="search-btn"
              >
                {loading ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Searching...</>
                ) : (
                  <><Search className="w-4 h-4 mr-2" /> Search Providers</>
                )}
              </Button>
              <Button 
                onClick={() => setShowChat(!showChat)} 
                variant="outline"
                data-testid="ai-chat-btn"
              >
                <MessageSquare className="w-4 h-4 mr-2" />
                AI Assistant
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        {providers.length > 0 && (
          <div data-testid="results-section">
            <h2 className="text-2xl font-bold mb-6">Available Providers</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {providers.map((provider) => (
                <Card key={provider.place_id} className="hover:shadow-xl transition-shadow" data-testid="provider-card">
                  {provider.photo_url && (
                    <img src={provider.photo_url} alt={provider.name} className="w-full h-40 object-cover rounded-t-lg" />
                  )}
                  <CardHeader>
                    <CardTitle className="text-lg">{provider.name}</CardTitle>
                    <CardDescription className="flex items-start gap-1">
                      <MapPin className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>{provider.address}</span>
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {provider.rating && (
                        <div className="flex items-center gap-2">
                          <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                          <span className="font-semibold">{provider.rating}</span>
                          {provider.user_ratings_total && (
                            <span className="text-sm text-gray-500">({provider.user_ratings_total} reviews)</span>
                          )}
                        </div>
                      )}
                      {provider.price_level && (
                        <div className="flex items-center gap-2">
                          <DollarSign className="w-4 h-4 text-green-600" />
                          <span>{"$".repeat(provider.price_level)}</span>
                        </div>
                      )}
                      {provider.phone && (
                        <div className="flex items-center gap-2 text-sm">
                          <Phone className="w-4 h-4 text-gray-600" />
                          <span>{provider.phone}</span>
                        </div>
                      )}
                      {provider.opening_hours && (
                        <Badge variant={provider.opening_hours === "Open" ? "default" : "secondary"}>
                          <Clock className="w-3 h-3 mr-1" />
                          {provider.opening_hours}
                        </Badge>
                      )}
                    </div>
                    <div className="mt-4 flex gap-2">
                      <Button 
                        onClick={() => {
                          setSelectedProvider(provider);
                          setShowBooking(true);
                        }}
                        className="flex-1"
                        data-testid="book-now-btn"
                      >
                        <CalendarIcon className="w-4 h-4 mr-2" />
                        Book Now
                      </Button>
                      {provider.website && (
                        <Button 
                          variant="outline" 
                          size="icon"
                          onClick={() => window.open(provider.website, '_blank')}
                          data-testid="visit-website-btn"
                        >
                          <Globe className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Booking Modal */}
      <Dialog open={showBooking} onOpenChange={setShowBooking}>
        <DialogContent data-testid="booking-modal">
          <DialogHeader>
            <DialogTitle>Book Appointment</DialogTitle>
            <DialogDescription>{selectedProvider?.name}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>Select Date</Label>
              <Calendar
                mode="single"
                selected={bookingDate}
                onSelect={setBookingDate}
                className="rounded-md border"
                data-testid="booking-calendar"
              />
            </div>
            <div>
              <Label htmlFor="time">Preferred Time</Label>
              <Input 
                id="time" 
                type="time" 
                value={bookingTime}
                onChange={(e) => setBookingTime(e.target.value)}
                data-testid="booking-time-input"
              />
            </div>
            <div>
              <Label htmlFor="notes">Notes (Optional)</Label>
              <Textarea 
                id="notes" 
                placeholder="Any special requirements?"
                value={bookingNotes}
                onChange={(e) => setBookingNotes(e.target.value)}
                data-testid="booking-notes-input"
              />
            </div>
            <Button onClick={handleBooking} className="w-full" data-testid="confirm-booking-btn">
              Confirm Booking
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Chat Panel */}
      {showChat && (
        <div className="fixed bottom-4 right-4 w-96 h-[500px] bg-white rounded-lg shadow-2xl border border-gray-200 flex flex-col z-50" data-testid="chat-panel">
          <div className="bg-blue-600 text-white p-4 rounded-t-lg flex justify-between items-center">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-5 h-5" />
              <span className="font-semibold">AI Assistant</span>
            </div>
            <Button variant="ghost" size="icon" onClick={() => setShowChat(false)} className="text-white hover:bg-blue-700">
              <X className="w-4 h-4" />
            </Button>
          </div>
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              {chatMessages.length === 0 && (
                <div className="text-center text-gray-500 py-8">
                  <MessageSquare className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                  <p>Ask me anything about local services!</p>
                </div>
              )}
              {chatMessages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] p-3 rounded-lg ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-900'}`}>
                    {msg.content}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 p-3 rounded-lg">
                    <Loader2 className="w-4 h-4 animate-spin" />
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
          <div className="p-4 border-t">
            <div className="flex gap-2">
              <Input 
                placeholder="Type your message..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleChatSend()}
                data-testid="chat-input"
              />
              <Button onClick={handleChatSend} disabled={chatLoading} data-testid="chat-send-btn">
                Send
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;