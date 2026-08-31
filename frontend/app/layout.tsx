import type { Metadata } from 'next';
import localFont from 'next/font/local';
import './globals.css';

// Fonts are vendored rather than fetched from Google at build time. `next/font/google`
// downloads during the build, so a slow or offline network fails the build outright —
// which is exactly the situation you are in when rebuilding on conference wifi before
// a demo. These are the same variable fonts, self-hosted.
const playfair = localFont({
  src: './fonts/PlayfairDisplay-Variable.woff2',
  variable: '--font-display',
  weight: '400 700',
  display: 'swap',
});

const inter = localFont({
  src: './fonts/Inter-Variable.woff2',
  variable: '--font-body',
  weight: '300 700',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'FinVoice',
  description: 'Raw bank calls to structured, auditable, ML-trainable data',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${playfair.variable} ${inter.variable}`}>
        {children}
      </body>
    </html>
  );
}
