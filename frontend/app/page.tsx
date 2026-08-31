import Navbar from '@/components/landing/Navbar';
import Hero from '@/components/landing/Hero';
import Pipeline from '@/components/landing/Pipeline';
import { Features, IntelligenceLayer, Infrastructure, CTAFooter } from '@/components/landing/Sections';
import ScrollAnimator from '@/components/landing/ScrollAnimator';

export default function LandingPage() {
  return (
    <div>
      <Navbar />
      <Hero />
      <Pipeline />
      <Features />
      <IntelligenceLayer />
      <Infrastructure />
      <CTAFooter />
      <ScrollAnimator />
    </div>
  );
}
