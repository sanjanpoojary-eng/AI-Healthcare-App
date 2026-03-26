import { SignIn, useUser } from '@clerk/nextjs';
import { useRouter } from 'next/router';
import { useEffect } from 'react';

export default function Home() {
  const { isSignedIn } = useUser();
  const router = useRouter();

  useEffect(() => {
    if (isSignedIn) {
      router.push('/dashboard');
    }
  }, [isSignedIn, router]);

  return (
    <div className="page-bg">
      <div className="glass-card">
        <h1>Personalized Healthcare System</h1>
        <SignIn
          routing="hash"
          signUpUrl="/sign-up"
          appearance={{
            variables: {
              colorPrimary: '#ff0080', // pinkish accent
            },
          }}
        />
      </div>
    </div>
  );
}
