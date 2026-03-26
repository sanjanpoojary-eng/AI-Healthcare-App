import Link from 'next/link';
import { useUser, UserButton, SignOutButton } from '@clerk/nextjs';
import { useRouter } from 'next/router';
import { useEffect } from 'react';

export default function Dashboard() {
  const { isSignedIn } = useUser();
  const router = useRouter();

  useEffect(() => {
    if (!isSignedIn) {
      router.push('/');
    }
  }, [isSignedIn, router]);

  return (
    <div className="page-bg">
      <header className="glass-header">
        <h1>Dashboard</h1>
        <div className="user-actions">
          <UserButton />
          <SignOutButton />
        </div>
      </header>
      <main className="glass-main">
        <div className="card">
          <h2>Disease Prediction</h2>
          <p>
            Predict diseases and generate detailed descriptions using ML + GenAI.
          </p>
          <Link href="http://127.0.0.1:5000" legacyBehavior>
            <a target="_blank" rel="noopener noreferrer">Go to Module</a>
          </Link>
        </div>
        <div className="card">
          <h2>Health Report Analyzer</h2>
          <p>
            Analyze health reports using Retrieval-Augmented Generation (RAG).
          </p>
          <Link href="http://localhost:8501" legacyBehavior>
            <a target="_blank" rel="noopener noreferrer">Go to Module</a>
          </Link>
        </div>
      </main>
    </div>
  );
}
