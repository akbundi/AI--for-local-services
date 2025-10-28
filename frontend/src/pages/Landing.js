import React, { useState, useContext } from 'react';
import { AuthContext } from '@/App';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import axios from 'axios';
import { API } from '@/App';
import { MapPin, Wrench, GraduationCap, Dumbbell, HardHat, Sparkles } from 'lucide-react';
import { toast } from 'sonner';

const Landing = () => {
  const { login } = useContext(AuthContext);
  const [showAuth, setShowAuth] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleAuth = async (isLogin, formData) => {
    setLoading(true);
    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register';
      const response = await axios.post(`${API}${endpoint}`, formData);
      login(response.data.access_token, response.data.user);
      toast.success(isLogin ? 'Welcome back!' : 'Account created successfully!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const handleLoginSubmit = (e) => {
    e.preventDefault();
    const formData = {
      email: e.target.email.value,
      password: e.target.password.value,
    };
    handleAuth(true, formData);
  };

  const handleRegisterSubmit = (e) => {
    e.preventDefault();
    const formData = {
      email: e.target.email.value,
      username: e.target.username.value,
      password: e.target.password.value,
    };
    handleAuth(false, formData);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="container mx-auto px-4 py-6 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <MapPin className="w-8 h-8 text-blue-600" />
          <span className="text-2xl font-bold text-gray-900">LocalProHelper</span>
        </div>
        <Button 
          onClick={() => setShowAuth(true)} 
          className="bg-blue-600 hover:bg-blue-700 text-white"
          data-testid="header-get-started-btn"
        >
          Get Started
        </Button>
      </header>

      {!showAuth ? (
        <main className="container mx-auto px-4 py-20">
          {/* Hero Section */}
          <div className="max-w-4xl mx-auto text-center mb-20">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6 leading-tight">
              Find Local Service Providers
              <span className="block text-blue-600 mt-2">In Seconds</span>
            </h1>
            <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
              AI-powered platform to discover, compare, and book trusted local professionals.
              From plumbers to tutors, we've got you covered.
            </p>
            <Button 
              size="lg" 
              onClick={() => setShowAuth(true)}
              className="bg-blue-600 hover:bg-blue-700 text-white text-lg px-8 py-6 rounded-full shadow-lg hover:shadow-xl transition-all"
              data-testid="hero-get-started-btn"
            >
              <Sparkles className="w-5 h-5 mr-2" />
              Start Finding Pros
            </Button>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            <FeatureCard
              icon={<Wrench className="w-10 h-10 text-blue-600" />}
              title="Plumbers"
              description="Emergency repairs, installations, and maintenance"
            />
            <FeatureCard
              icon={<GraduationCap className="w-10 h-10 text-green-600" />}
              title="Tutors"
              description="Expert tutoring for all subjects and levels"
            />
            <FeatureCard
              icon={<Dumbbell className="w-10 h-10 text-purple-600" />}
              title="Gyms & Fitness"
              description="Find the perfect gym or personal trainer"
            />
            <FeatureCard
              icon={<HardHat className="w-10 h-10 text-orange-600" />}
              title="Repair Services"
              description="Home repairs, electronics, and appliances"
            />
          </div>

          {/* How It Works */}
          <div className="mt-32 max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-center text-gray-900 mb-16">How It Works</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <StepCard number="1" title="Enter Location" description="Tell us where you need services" />
              <StepCard number="2" title="AI Recommendations" description="Get personalized suggestions based on your budget" />
              <StepCard number="3" title="Book Instantly" description="Schedule appointments with ease" />
            </div>
          </div>
        </main>
      ) : (
        <main className="container mx-auto px-4 py-20 flex items-center justify-center">
          <Card className="w-full max-w-md shadow-xl" data-testid="auth-card">
            <CardHeader>
              <CardTitle className="text-2xl">Welcome</CardTitle>
              <CardDescription>Sign in or create an account to continue</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="login" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="login" data-testid="login-tab">Login</TabsTrigger>
                  <TabsTrigger value="register" data-testid="register-tab">Register</TabsTrigger>
                </TabsList>
                <TabsContent value="login">
                  <form onSubmit={handleLoginSubmit} className="space-y-4" data-testid="login-form">
                    <div>
                      <Label htmlFor="email">Email</Label>
                      <Input id="email" name="email" type="email" required data-testid="login-email-input" />
                    </div>
                    <div>
                      <Label htmlFor="password">Password</Label>
                      <Input id="password" name="password" type="password" required data-testid="login-password-input" />
                    </div>
                    <Button type="submit" className="w-full" disabled={loading} data-testid="login-submit-btn">
                      {loading ? 'Loading...' : 'Login'}
                    </Button>
                  </form>
                </TabsContent>
                <TabsContent value="register">
                  <form onSubmit={handleRegisterSubmit} className="space-y-4" data-testid="register-form">
                    <div>
                      <Label htmlFor="username">Username</Label>
                      <Input id="username" name="username" required data-testid="register-username-input" />
                    </div>
                    <div>
                      <Label htmlFor="reg-email">Email</Label>
                      <Input id="reg-email" name="email" type="email" required data-testid="register-email-input" />
                    </div>
                    <div>
                      <Label htmlFor="reg-password">Password</Label>
                      <Input id="reg-password" name="password" type="password" required data-testid="register-password-input" />
                    </div>
                    <Button type="submit" className="w-full" disabled={loading} data-testid="register-submit-btn">
                      {loading ? 'Loading...' : 'Create Account'}
                    </Button>
                  </form>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </main>
      )}

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 text-center text-gray-600 border-t border-gray-200 mt-20">
        <p>&copy; 2025 LocalProHelper. Find trusted local professionals near you.</p>
      </footer>
    </div>
  );
};

const FeatureCard = ({ icon, title, description }) => (
  <Card className="text-center hover:shadow-lg transition-shadow border-2 hover:border-blue-200">
    <CardContent className="pt-6">
      <div className="mb-4 flex justify-center">{icon}</div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-600 text-sm">{description}</p>
    </CardContent>
  </Card>
);

const StepCard = ({ number, title, description }) => (
  <div className="text-center">
    <div className="w-16 h-16 bg-blue-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
      {number}
    </div>
    <h3 className="text-xl font-semibold mb-2">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </div>
);

export default Landing;